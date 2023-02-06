# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.losses.utils import weighted_loss
from ...core.bbox import AxisAlignedBboxOverlaps3D
from ..builder import LOSSES


@weighted_loss
def axis_aligned_iou_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).
        target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou
    return iou_loss


@weighted_loss
def axis_aligned_diou_loss(pred, target):
    """Calculate the DIoU loss (1-DIoU) of two sets of axis aligned bounding
    boxes. Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).
        target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
            (x1, y1, z1, x2, y2, z2).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    axis_aligned_iou = AxisAlignedBboxOverlaps3D()(
        pred, target, is_aligned=True)
    iou_loss = 1 - axis_aligned_iou

    xp1, yp1, zp1, xp2, yp2, zp2 = pred.split(1, dim=-1)
    xt1, yt1, zt1, xt2, yt2, zt2 = target.split(1, dim=-1)

    xpc = (xp1 + xp2) / 2
    ypc = (yp1 + yp2) / 2
    zpc = (zp1 + zp2) / 2
    xtc = (xt1 + xt2) / 2
    ytc = (yt1 + yt2) / 2
    ztc = (zt1 + zt2) / 2
    r2 = (xpc - xtc) ** 2 + (ypc - ytc) ** 2 + (zpc - ztc) ** 2

    x_min = torch.minimum(xp1, xt1)
    x_max = torch.maximum(xp2, xt2)
    y_min = torch.minimum(yp1, yt1)
    y_max = torch.maximum(yp2, yt2)
    z_min = torch.minimum(zp1, zt1)
    z_max = torch.maximum(zp2, zt2)
    c2 = (x_min - x_max) ** 2 + (y_min - y_max) ** 2 + (z_min - z_max) ** 2

    diou_loss = iou_loss + (r2 / c2)[:, 0]

    return diou_loss


@LOSSES.register_module()
class AxisAlignedIoULoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of axis aligned bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, mode='iou', reduction='mean', loss_weight=1.0):
        super(AxisAlignedIoULoss, self).__init__()
        self.loss = axis_aligned_iou_loss if mode == 'iou' else axis_aligned_diou_loss
        assert reduction in ['none', 'sum', 'mean']
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
            pred (torch.Tensor): Bbox predictions with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 6]
                (x1, y1, z1, x2, y2, z2).
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
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        return self.loss(
            pred,
            target,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction) * self.loss_weight
