import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward
from pytorch_wavelets.dwt.transform2d import SWTForward
from .two_stage import TwoStageDetector
from ..builder import DETECTORS

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def map_boxes_to_feat_level(gt_bboxes, feat_hw, img_meta):
    Hf, Wf = feat_hw
    Hp, Wp = img_meta.get('pad_shape', img_meta['img_shape'])[:2]
    scale_x = Wf / float(Wp)
    scale_y = Hf / float(Hp)

    mapped = []
    for boxes in gt_bboxes:
        if boxes is None or boxes.numel() == 0:
            mapped.append(
                torch.zeros((0, 4), device=boxes.device if boxes is not None else 'cpu')
            )
        else:
            b = boxes.clone().float()
            b[:, [0, 2]] *= scale_x
            b[:, [1, 3]] *= scale_y
            mapped.append(b)
    return mapped


def boxes_to_binary_mask(boxes, H, W, device):
    m = torch.zeros(1, 1, H, W, device=device)
    if boxes is None or boxes.numel() == 0:
        return m
    for x1, y1, x2, y2 in boxes:
        x1i = max(int(torch.floor(x1).item()), 0)
        y1i = max(int(torch.floor(y1).item()), 0)
        x2i = min(int(torch.ceil(x2).item()), W - 1)
        y2i = min(int(torch.ceil(y2).item()), H - 1)
        if x2i >= x1i and y2i >= y1i:
            m[:, :, y1i:y2i + 1, x1i:x2i + 1] = 1.0
    return m


def bce_with_pos_weight(logit, target, min_pw=1.0, max_pw=20.0, eps=1e-6, reduction='mean'):
    B, _, H, W = target.shape
    with torch.no_grad():
        pos = target.view(B, -1).sum(dim=1) + eps
        tot = torch.tensor(H * W, device=target.device, dtype=target.dtype)
        neg = tot - pos
        pw = torch.clamp(neg / pos, min=min_pw, max=max_pw).view(B, 1, 1, 1)
    loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw, reduction=reduction)
    return loss

def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape

    pad = kernel_size // 2
    # print(pad)
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    # print(pad_x.shape)
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
    unfold_x = unfold_x[..., :m_h, :m_w].reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    # test
    # res[:, :, 0] = 1
    # res[:, :, 1] = 2
    # res[:, :, 2] = 3
    # res[:, :, 3] = 4
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    # res = F.pixel_shuffle(res, up)
    # print(res.shape)
    # print(res)
    return res

class PredAlignUpsample(nn.Module):
    def __init__(self, feat_channels, ksize=5, groups=1, compress_channels=64                 ,
                 pad_mode_feat="replicate", pad_mode_logit="constant", pad_value_logit=0.0):
        super().__init__()
        self.ksize = ksize
        self.groups = groups
        self.pad_mode_feat = pad_mode_feat
        self.pad_mode_logit = pad_mode_logit
        self.pad_value_logit = pad_value_logit
        self.hr_compress = nn.Conv2d(feat_channels, compress_channels, 1)
        self.content_encoder = nn.Conv2d(
            compress_channels, ksize * ksize * groups, kernel_size=3, padding=1
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.hr_compress.weight); nn.init.zeros_(self.hr_compress.bias)
        nn.init.normal_(self.content_encoder.weight, std=1e-3); nn.init.zeros_(self.content_encoder.bias)

    @staticmethod
    def _lcm(a, b):
        return a * b // math.gcd(a, b)

    @staticmethod
    def _pad_to(x, H2, W2, mode="replicate", value=0.0):
        H, W = x.shape[-2:]
        pad_h = H2 - H
        pad_w = W2 - W
        if pad_h == 0 and pad_w == 0:
            return x
        if mode == "constant":
            return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=value)
        else:
            return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    @staticmethod
    def _kernel_normalizer(mask, k):
        # mask: [B, G*k*k, H, W] -> 按每个位置上的 k*k 做 softmax + 归一化
        B, C, H, W = mask.size()
        G = C // (k * k)
        mask = mask.view(B, G, k * k, H, W)
        mask = F.softmax(mask, dim=2).view(B, G * k * k, H, W).contiguous()
        return mask

    def forward(self, pred, feat):
        B, _, H, W = feat.shape
        _, Cp, h, w = pred.shape

        up_h = math.ceil(H / float(h))
        up_w = math.ceil(W / float(w))
        up = max(up_h, up_w)

        guide = self.hr_compress(feat)
        raw_mask = self.content_encoder(guide)
        normed_mask = self._kernel_normalizer(raw_mask, self.ksize)

        pred_up = carafe(pred, normed_mask, kernel_size=self.ksize, group=self.groups, up=up)
        return pred_up

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.bn.train(True)
        self.bn.track_running_stats = False
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Supervision_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Supervision_Layer, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.ReLU()

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class WH(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.wt = SWTForward(J=1, wave='haar',mode='symmetric')
        self.csp = CSPRepLayer(in_ch * 3, out_ch)

    def forward(self, x):
        yh_list = self.wt(x)

        bands = yh_list[0]

        if bands.dim() == 4:
            B, C4, H, W = bands.shape
            assert C4 % 4 == 0
            C = C4 // 4
            bands = bands.view(B, C, 4, H, W)
        elif bands.dim() == 5:
            pass
        else:
            raise RuntimeError(f'unexpected SWT shape: {bands.shape}')

        Hb = bands[:, :, 1, :, :]
        Vb = bands[:, :, 2, :, :]
        Db = bands[:, :, 3, :, :]
        yH = torch.cat([Hb, Vb, Db], dim=1)

        yH = self.csp(yH)
        return yH + x

class WL(nn.Module): #小波变化高低频分解模块
    def __init__(self, in_ch, out_ch):
        super(WL, self).__init__()
        self.wt = SWTForward(J=1, mode='symmetric', wave='haar')
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(2*in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y_list = self.wt(x)
        bands = y_list[0]
        B, C4, H, W = bands.shape
        assert C4 % 4 == 0
        C = C4 // 4
        bands = bands.view(B, C, 4, H, W)
        yL = bands[:, :, 0, :, :]
        return x + yL

class ADWF(nn.Module):
    def __init__(self, in_c):
        super(ADWF, self).__init__()
        self.WH = WH(in_c, 256)
        self.supervision_layer = Supervision_Layer(256)
        self.pred_up = PredAlignUpsample(feat_channels=256, ksize=5)
        self.pos_head = nn.Conv2d(256, 1, kernel_size=1, padding=0,bias=True)


    def forward(self, feat, pred):
        pred_up = self.pred_up(pred, feat)
        wh_in = feat * pred_up
        wh = self.WH(wh_in)
        wh = self.supervision_layer(wh)
        out = wh + feat
        pos_logit = self.pos_head(out)
        return out, pos_logit, pred_up


@DETECTORS.register_module()
class CascadeRCNN_DFEDet(TwoStageDetector):
    def __init__(self,
                 backbone, neck=None, rpn_head=None, roi_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None,
                 pos_guided_weight=1,
                 pos_loss_type='bce',
                 focal_alpha=0.5,
                 focal_gamma=2.0,
                 bce_min_pw=1.0,
                 bce_max_pw=20.0,
                 use_pos_head=True
                 ):
        super(CascadeRCNN_DFEDet, self).__init__(
            backbone=backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head,
            train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained, init_cfg=init_cfg)

        self.wl = WL(256, 256)
        self.adwf = ADWF(256)
        self.pos_guided_weight = pos_guided_weight
        self.pos_loss_type = pos_loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce_min_pw = bce_min_pw
        self.bce_max_pw = bce_max_pw
        self.use_pos_head = use_pos_head
        self._last_pos_logit = None
        self._last_pos_stride_meta = None

    def extract_feat(self, img, img_metas):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x1 = list(x)

        p4 = self.wl(x1[4])
        x1_0, pos_logit, pred_up = self.adwf(x1[0], p4)
        x1[0] = x1_0

        self._last_pos_logit = pos_logit
        self._last_pos_stride_meta = {
            'feat_hw': pos_logit.shape[-2:],
            'img_meta0': img_metas[0],
        }
        return x1

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs):
        x = self.extract_feat(img, img_metas)
        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        # DASB
        if self._last_pos_logit is not None and self.pos_guided_weight > 0:
            pos_logit = self._last_pos_logit  # [B,1,Hf,Wf]
            B, _, Hf, Wf = pos_logit.shape

            masks = []
            for i in range(B):
                mapped_i = map_boxes_to_feat_level(
                    [gt_bboxes[i]], (Hf, Wf), img_metas[i]
                )[0]

                m = boxes_to_binary_mask(mapped_i, Hf, Wf, pos_logit.device)  # [1,1,Hf,Wf]
                masks.append(m)

            tgt_mask = torch.cat(masks, dim=0)  # [B,1,Hf,Wf]

            loss_pos = bce_with_pos_weight(
                pos_logit, tgt_mask,
                min_pw=self.bce_min_pw, max_pw=self.bce_max_pw
            )

            losses['loss_pos_guided'] = loss_pos * self.pos_guided_weight

            self._last_pos_logit = None
            self._last_pos_stride_meta = None

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img, img_metas)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_DFEDet, self).show_result(data, result, **kwargs)

