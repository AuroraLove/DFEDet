# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

# from .. import BaseDetector
# from ..builder import MODELS
# from .detr import DETR


# @MODELS.register_module()
class RTDETR():

    def __init__(self, encoder,decoder, criterion,*args, **kwargs):
        super(BaseModule, self).__init__()
        self.encoder = encoder  # 包装 RTDETR
        self.decoder = decoder  # 包装 RTDETR
        self.criterion = criterion  # 使用 RTDETRCriterion

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        x = self.encoder(img)
        x = self.decoder(x)
        metas = dict(epoch=epoch, step=i, global_step=global_step)  # 元信息传递

        losses = self.criterion(x, targets, **metas)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses