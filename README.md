[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=tr3d-towards-real-time-indoor-3d-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-sun-rgbd-val)](https://paperswithcode.com/sota/3d-object-detection-on-sun-rgbd-val?p=tr3d-towards-real-time-indoor-3d-object)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tr3d-towards-real-time-indoor-3d-object/3d-object-detection-on-s3dis)](https://paperswithcode.com/sota/3d-object-detection-on-s3dis?p=tr3d-towards-real-time-indoor-3d-object)

## TR3D: Towards Real-Time Indoor 3D Object Detection

**News**:
 * :fire: June, 2023. TR3D is accepted at [ICIP2023](https://2023.ieeeicip.org/).
 * :rocket: June, 2023. We add ScanNet-pretrained S3DIS model and log significantly pushing forward state-of-the-art.
 * February, 2023. TR3D on all 3 datasets is now supported in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) as a [project](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/TR3D).
 * :fire: February, 2023. TR3D is now state-of-the-art on [paperswithcode](https://paperswithcode.com) on SUN RGB-D and S3DIS.

This repository contains an implementation of TR3D, a 3D object detection method introduced in our paper:

> **TR3D: Towards Real-Time Indoor 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung AI Center Moscow <br>
> https://arxiv.org/abs/2302.02858

### Installation
For convenience, we provide a [Dockerfile](docker/Dockerfile).

Alternatively, you can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [getting_started.md](docs/en/getting_started.md), including MinkowskiEngine installation, replacing `open-mmlab/mmdetection3d` with `samsunglabs/tr3d`.


Most of the `TR3D`-related code locates in the following files: 
[detectors/mink_single_stage.py](mmdet3d/models/detectors/mink_single_stage.py),
[detectors/tr3d_ff.py](mmdet3d/models/detectors/tr3d_ff.py),
[dense_heads/tr3d_head.py](mmdet3d/models/dense_heads/tr3d_head.py),
[necks/tr3d_neck.py](mmdet3d/models/necks/tr3d_neck.py).

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We follow the mmdetection3d data preparation protocol described in [scannet](data/scannet), [sunrgbd](data/sunrgbd), and [s3dis](data/s3dis).

**Training**

To start training, run [train](tools/train.py) with TR3D [configs](configs/tr3d):
```shell
python tools/train.py configs/tr3d/tr3d_scannet-3d-18class.py
```

**Testing**

Test pre-trained model using [test](tools/dist_test.sh) with TR3D [configs](configs/tr3d):
```shell
python tools/test.py configs/tr3d/tr3d_scannet-3d-18class.py \
    work_dirs/tr3d_scannet-3d-18class/latest.pth --eval mAP
```

**Visualization**

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.3`:
```shell
python tools/test.py configs/tr3d/tr3d_scannet-3d-18class.py \
    work_dirs/tr3d_scannet-3d-18class/latest.pth --eval mAP --show \
    --show-dir work_dirs/tr3d_scannet-3d-18class
```

### Models

The metrics are obtained in 5 training runs followed by 5 test runs. We report both the best and the average values (the latter are given in round brackets).
Inference speed (scenes per second) is measured on a single NVidia RTX 4090. Please, note that ScanNet-pretrained S3DIS model was actually trained in the original
[openmmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/TR3D) codebase.

**TR3D 3D Detection**

| Dataset | mAP@0.25 | mAP@0.5 | Scenes <br> per sec.| Download |
|:-------:|:--------:|:-------:|:-------------------:|:--------:|
| ScanNet | 72.9 (72.0) | 59.3 (57.4) | 23.7 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet.log.json) &#124; [config](configs/tr3d/tr3d_scannet-3d-18class.py) |
| SUN RGB-D | 67.1 (66.3) | 50.4 (49.6) | 27.5 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_sunrgbd.log.json) &#124; [config](configs/tr3d/tr3d_sunrgbd-3d-10class.py) |
| S3DIS | 74.5 (72.1) | 51.7 (47.6) | 21.0 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_s3dis.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_s3dis.log.json) &#124; [config](configs/tr3d/tr3d_s3dis-3d-5class.py) |
| S3DIS <br> ScanNet-pretrained | 75.9 (75.1) | 56.6 (54.8) | 21.0 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet-pretrain_s3dis.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_scannet-pretrain_s3dis.log) &#124; [config](configs/tr3d/tr3d_scannet-pretrain_s3dis-3d-5class.py) |

**RGB + PC 3D Detection on SUN RGB-D**

| Model | mAP@0.25 | mAP@0.5 | Scenes <br> per sec.| Download |
|:-----:|:--------:|:-------:|:-------------------:|:--------:|
| ImVoteNet | 63.4 | - | 14.8 | [instruction](configs/imvotenet) |
| VoteNet+FF | 64.5 (63.7) | 39.2 (38.1) | - | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/votenet_ff_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/votenet_ff_sunrgbd.log.json) &#124; [config](configs/votenet/votenet-ff_16x8_sunrgbd-3d-10class.py) |
| TR3D+FF | 69.4 (68.7) | 53.4 (52.4) | 17.5 | [model](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_ff_sunrgbd.pth) &#124; [log](https://github.com/samsunglabs/tr3d/releases/download/v1.0/tr3d_ff_sunrgbd.log.json) &#124; [config](configs/tr3d/tr3d-ff_sunrgbd-3d-10class.py) |

### Example Detections

<p align="center"><img src="./resources/github.png" alt="drawing" width="90%"/></p>

### Citation

If you find this work useful for your research, please cite our paper:

```
@misc{rukhovich2023tr3d,
  doi = {10.48550/ARXIV.2302.02858},
  url = {https://arxiv.org/abs/2302.02858},
  author = {Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  title = {TR3D: Towards Real-Time Indoor 3D Object Detection},
  publisher = {arXiv},
  year = {2023}
}
```
