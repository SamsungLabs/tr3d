_base_ = ['./tr3d_s3dis-3d-5class.py']

load_from = 'work_dirs/tmp/tr3d_scannet.pth'

optim_wrapper = dict(
    optimizer=dict(weight_decay=0.001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
