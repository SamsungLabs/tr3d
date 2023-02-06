voxel_size = .01
n_points = 100000

model = dict(
    type='TR3DFF3DDetector',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    backbone=dict(
        type='MinkFFResNet',
        in_channels=3,
        max_channels=128,
        depth=34,
        norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=10,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none')),
    voxel_size=voxel_size,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/imvotenet/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth'  # noqa
resume_from = None
workflow = [('train', 1)]


dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=True, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=True, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
