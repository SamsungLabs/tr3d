# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from functools import partial

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS, build_backbone, build_neck
from .single_stage import SingleStage3DDetector
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type


@DETECTORS.register_module()
class VoteNet(SingleStage3DDetector):
    r"""`VoteNet <https://arxiv.org/pdf/1904.09664.pdf>`_ for 3D detection."""

    def __init__(self,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNet, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
            bbox_list = self.bbox_head.get_bboxes(
                pts_cat, bbox_preds, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]


@DETECTORS.register_module()
class VoteNetFF(VoteNet):
    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoteNetFF, self).__init__(
            backbone=backbone,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=None,
            pretrained=pretrained)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
    
    def init_weights(self, pretrained=None):
        # self.img_backbone.init_weights()
        # self.img_neck.init_weights()
        for param in self.img_backbone.parameters():
            param.requires_grad = False
        for param in self.img_neck.parameters():
            param.requires_grad = False
        self.img_backbone.eval()
        self.img_neck.eval()
        self.backbone.init_weights()
        self.bbox_head.init_weights()
    
    def _f(self, xyz, features, img_features, img_metas, img_shape):
        projected_features = []
        for point, img_feature, img_meta in zip(xyz, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))

        projected_features = torch.cat(projected_features, dim=0)
        projected_features = projected_features.transpose(0, 1).unsqueeze(0)
        projected_features = self.conv(projected_features)
        batch_size, n_features, n_points = features.shape
        projected_features = projected_features[0].reshape(
            n_features, batch_size, n_points).transpose(0, 1)
        return projected_features + features
    
    def extract_feat(self, points, img, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features = self.img_neck(x)[0]

        x = self.backbone(points, partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, imgs, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img, img_meta)
            for pts, img, img_meta in zip(points, imgs, img_metas)
        ]

    def forward_train(self,
                      points,
                      img,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(
            bbox_preds, *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, points, img_metas, img, imgs=None, rescale=False):
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat, img, img_metas)
        bbox_preds = self.bbox_head(x, self.test_cfg.sample_mod)
        bbox_list = self.bbox_head.get_bboxes(
            points_cat, bbox_preds, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results