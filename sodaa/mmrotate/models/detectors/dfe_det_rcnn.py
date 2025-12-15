# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .two_stage import RotatedTwoStageDetector
from ..builder import ROTATED_DETECTORS

from pytorch_wavelets import DWTForward
from pytorch_wavelets.dwt.transform2d import SWTForward

def bce_with_pos_weight(logit, target, min_pw=1.0, max_pw=20.0, eps=1e-6, reduction='mean'):
    """
    logit: [B,1,H,W]
    target: [B,1,H,W] in {0,1}
    pos_weight 随 batch 自适应：neg/pos，限制到 [min_pw, max_pw]
    """
    B, _, H, W = target.shape
    with torch.no_grad():
        pos = target.view(B, -1).sum(dim=1) + eps
        tot = torch.tensor(H * W, device=target.device, dtype=target.dtype)
        neg = tot - pos
        pw = torch.clamp(neg / pos, min=min_pw, max=max_pw).view(B, 1, 1, 1)
    # PyTorch 的 pos_weight 是按样本张量 broadcast 的
    loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pw, reduction=reduction)
    return loss

def map_obbs_to_feat_level(gt_obbs, feat_hw, img_meta):
    Hf, Wf = feat_hw
    Hp, Wp = img_meta.get('pad_shape', img_meta['img_shape'])[:2]
    sx, sy = Wf / float(Wp), Hf / float(Hp)  # 一般 sx==sy（FPN 等距下采样）

    mapped = []
    for obb in gt_obbs:
        if obb is None or obb.numel() == 0:
            mapped.append(torch.zeros((0, 5), device=(obb.device if obb is not None else 'cpu')))
        else:
            b = obb.clone().float()
            b[:, 0] *= sx  # cx
            b[:, 1] *= sy  # cy
            b[:, 2] *= sx  # w
            b[:, 3] *= sy  # h
            mapped.append(b)
    return mapped

def obbs_to_binary_mask(obbs, H, W, device, shrink=1.0, dilate=0):

    m = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)
    if obbs is None or obbs.numel() == 0:
        return m

    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    Y, X = torch.meshgrid(ys, xs, indexing='ij')  # H×W

    mask = torch.zeros(H, W, device=device, dtype=torch.bool)
    for cx, cy, w, h, a in obbs:
        w = w * shrink
        h = h * shrink
        ca, sa = torch.cos(a), torch.sin(a)
        Xc = X - cx
        Yc = Y - cy
        xr =  Xc * ca + Yc * sa
        yr = -Xc * sa + Yc * ca
        inside = (xr.abs() <= w * 0.5) & (yr.abs() <= h * 0.5)
        mask |= inside

    m[0, 0] = mask.float()

    if dilate > 0:
        k = 2 * dilate + 1
        m = torch.nn.functional.max_pool2d(m, kernel_size=k, stride=1, padding=dilate)
        m = (m > 0).float()
    return m

def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape

    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    unfold_x = unfold_x[..., :m_h, :m_w].reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
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
        # 生成 k*k*groups 个核系数（在 HR 空间生成）
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
        B, C, H, W = mask.size()
        G = C // (k * k)
        mask = mask.view(B, G, k * k, H, W)
        mask = F.softmax(mask, dim=2).view(B, G * k * k, H, W).contiguous()
        return mask

    def forward(self, pred, feat):
        B, _, H, W = feat.shape
        _, Cp, h, w = pred.shape

        up = math.ceil(H / h)  # = up_w
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


class WL(nn.Module):
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

class ADWF(nn.Module):
    def __init__(self, in_c):
        super(ADWF, self).__init__()
        self.WH = WH(in_c, 256)

        self.csp = CSPRepLayer(256,256)
        self.pred_up = PredAlignUpsample(feat_channels=256, ksize=5)

        self.pos_head = nn.Conv2d(256, 1, kernel_size=1, padding=0,bias=True)


    def forward(self, feat, pred):
        pred_up = self.pred_up(pred, feat)

        wh_in = feat * pred_up
        ffH = self.WH(wh_in)

        ef = self.csp(ffH)
        out = ef + feat
        pos_logit = self.pos_head(out)

        return out, pos_logit, pred_up

@ROTATED_DETECTORS.register_module()
class DFEDetRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 pos_guided_weight=0.5,
                 bce_min_pw=1.0,
                 bce_max_pw=20.0,
                 ):
        super(DFEDetRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.wl = WL(256, 256)
        self.adwf = ADWF(256)

        self.pos_guided_weight = pos_guided_weight
        self.bce_min_pw = bce_min_pw
        self.bce_max_pw = bce_max_pw


    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x1 = list(x)

        p4 = self.wl(x1[4])
        x1_0, pos_logit, pred_up = self.adwf(x1[0], p4)
        x1[0] = x1_0

        return x1,pos_logit

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x,pos_logit = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x,pos_logit = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # DASB
        if pos_logit is not None and self.pos_guided_weight > 0:
            B, _, Hf, Wf = pos_logit.shape
            masks = []
            for i in range(B):
                mapped_i = map_obbs_to_feat_level([gt_bboxes[i]], (Hf, Wf), img_metas[i])[0]
                m = obbs_to_binary_mask(mapped_i, Hf, Wf, pos_logit.device, shrink=0.9, dilate=1)
                masks.append(m)
            tgt_mask = torch.cat(masks, dim=0)  # [B,1,Hf,Wf]

            loss_pos = bce_with_pos_weight(
                pos_logit, tgt_mask,
                min_pw=self.bce_min_pw, max_pw=self.bce_max_pw
            )

            losses['loss_pos_guided'] = loss_pos * self.pos_guided_weight

        return losses


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x,pos_logit = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs
