# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import diff_iou_rotated_3d
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d
from torch import nn as nn

from mmdet.models.losses.utils import weighted_loss
from ..builder import LOSSES


def diff_diou_rotated_3d(box3d1, box3d2):
    """Calculate differentiable iou of rotated 3d boxes.

    Args:
        box3d1 (Tensor): (B, N, 3+3+1) First box (x,y,z,w,h,l,alpha).
        box3d2 (Tensor): (B, N, 3+3+1) Second box (x,y,z,w,h,l,alpha).

    Returns:
        Tensor: (B, N) IoU.
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) -
                 torch.max(zmin1, zmin2)).clamp_(min=0.)
    intersection_3d = intersection * z_overlap
    volume1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    volume2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    union_3d = volume1 + volume2 - intersection_3d

    x1_max = torch.max(corners1[..., 0], dim=2)[0]     # (B, N)
    x1_min = torch.min(corners1[..., 0], dim=2)[0]     # (B, N)
    y1_max = torch.max(corners1[..., 1], dim=2)[0]
    y1_min = torch.min(corners1[..., 1], dim=2)[0]
    
    x2_max = torch.max(corners2[..., 0], dim=2)[0]     # (B, N)
    x2_min = torch.min(corners2[..., 0], dim=2)[0]    # (B, N)
    y2_max = torch.max(corners2[..., 1], dim=2)[0]
    y2_min = torch.min(corners2[..., 1], dim=2)[0]

    x_max = torch.max(x1_max, x2_max)
    x_min = torch.min(x1_min, x2_min)
    y_max = torch.max(y1_max, y2_max)
    y_min = torch.min(y1_min, y2_min)

    z_max = torch.max(zmax1, zmax2)
    z_min = torch.min(zmin1, zmin2)

    r2 = ((box1[..., :3] - box2[..., :3]) ** 2).sum(dim=-1)
    c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2
    
    return intersection_3d / union_3d - r2 / c2


@weighted_loss
def rotated_diou_3d_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of rotated bounding boxes.
    Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [N, 7]
            (x, y, z, w, l, h, alpha).
        target (torch.Tensor): Bbox targets (gt) with shape [N, 7]
            (x, y, z, w, l, h, alpha).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    diou_loss = 1 - diff_diou_rotated_3d(pred.unsqueeze(0),
                                         target.unsqueeze(0))[0]
    return diou_loss


@weighted_loss
def rotated_iou_3d_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of rotated bounding boxes.
    Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [N, 7]
            (x, y, z, w, l, h, alpha).
        target (torch.Tensor): Bbox targets (gt) with shape [N, 7]
            (x, y, z, w, l, h, alpha).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    iou_loss = 1 - diff_iou_rotated_3d(pred.unsqueeze(0),
                                       target.unsqueeze(0))[0]
    return iou_loss


@LOSSES.register_module()
class RotatedIoU3DLoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of rotated bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, mode='iou', reduction='mean', loss_weight=1.0):
        super().__init__()
        self.loss = rotated_iou_3d_loss if mode == 'iou' else rotated_diou_3d_loss
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * self.loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss
