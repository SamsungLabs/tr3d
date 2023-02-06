# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from torch import nn
from functools import partial

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from mmdet3d.models.fusion_layers.point_fusion import point_sample
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from .base import Base3DDetector


@DETECTORS.register_module()
class TR3DFF3DDetector(Base3DDetector):
    r"""TR3D+FF Detector

    Args:
        img_backbone (dict): Config of the 2D backbone.
        img_neck (dict): Config of the 2D neck.
        backbone (dict): Config of the 3D backbone.
        neck (dict): Config of the 3D neck.
        head (dict): Config of the 3D head.
        voxel_size (float): Voxel size in meters.
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
        pretrained (str, optional): Deprecated initialization parameter.
            Defaults to None.
    """

    def __init__(self,
                 img_backbone,
                 img_neck,
                 backbone,
                 neck,
                 head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(TR3DFF3DDetector, self).__init__(init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(256, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))

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
        self.neck.init_weights()
        self.head.init_weights()

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
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
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        projected_features = self.conv(projected_features)
        return projected_features + x

    def extract_feat(self, *args):
        """Just implement @abstractmethod of BaseModule."""

    def extract_feats(self, points, img, img_metas):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        with torch.no_grad():
            x = self.img_backbone(img)
            img_features = self.img_neck(x)[0]
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p[:, 3:]) for p in points],
            device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)
        x = self.backbone(x, partial(
            self._f, img_features=img_features, img_metas=img_metas, img_shape=img.shape))
        x = self.neck(x)
        return x

    def forward_train(self, points, img, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels(list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Centerness, bbox and classification loss values.
        """
        x = self.extract_feats(points, img, img_metas)
        losses = self.head.forward_train(x, gt_bboxes_3d, gt_labels_3d,
                                         img_metas)
        return losses

    def simple_test(self, points, img_metas, img, *args, **kwargs):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        x = self.extract_feats(points, img, img_metas)
        bbox_list = self.head.forward_test(x, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
