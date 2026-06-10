
# FWDet: Frequency-Consistent Wavelet Learning for Detail-Preserving Aerial Small Object Detection

This repository provides the implementation of **FWDet**, a frequency-consistent wavelet learning framework for aerial small object detection. FWDet introduces a decomposition--reconstruction--fusion pipeline to explicitly model low-frequency semantic structures and high-frequency detail cues for UAV-based small object detection.

The code is associated with the manuscript:

**Frequency-Consistent Wavelet Learning for Detail-Preserving Aerial Small Object Detection**

## 1. Overview

FWDet is built based on the YOLOv10 framework. The main idea is to introduce wavelet-domain representation learning into hierarchical feature extraction and multi-scale feature fusion. Specifically, the proposed framework contains:

- **Freq-Decomp**: multi-head wavelet-based frequency decomposition using DWT;
- **Freq-Recon**: direction-aware frequency reconstruction using IDWT;
- **Freq-Fusion**: frequency-consistent cross-scale feature fusion;
- **YOLOv10 Detection Head**: the original YOLOv10 detection head is retained.

## 2. Environment

The running environment is the same as the official YOLOv10 repository. In addition, this project requires `pytorch_wavelets`.

### Main dependencies

```bash
python >= 3.8
torch
torchvision
ultralytics
opencv-python
numpy
pytorch_wavelets
````

Install the additional wavelet package:

```bash
pip install pytorch_wavelets
```

The key additional import used in FWDet is:

```python
from pytorch_wavelets import DWTForward, DWTInverse
```

## 3. Dataset Preparation

Please place the datasets under the `datasets/` directory.

The expected structure is:

```text
FWDet/
├── datasets/
│   ├── VisDrone2019/
│   │   ├── images/
│   │   ├── labels/
│   │   └── ...
│   └── UAVDT/
│       ├── images/
│       ├── labels/
│       └── ...
├── train.py
├── test.py
├── result.py
└── ...
```

Please make sure that the dataset configuration file correctly points to the training, validation, and test image paths.

## 4. Pretrained Weight

The trained weight file `best.pt` is provided through Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1C3Fsl0rx1TZ_xaiMUdduMw?pwd=sbkc
Extraction code: sbkc
```

After downloading, please place `best.pt` in a proper directory, for example:

```text
FWDet/weights/best.pt
```

## 5. Training

To train FWDet, run:

```bash
python train.py
```

Before training, please check the dataset path, model configuration, image size, batch size, device ID, and other training settings in `train.py`.

## 6. Testing

To evaluate the model on the test set, run:

```bash
python test.py
```

Please make sure that the weight path in `test.py` points to the downloaded or trained model weight, such as:

```text
weights/best.pt
```

## 7. Inference on Images

To test one image or several images, run:

```bash
python result.py
```

Before running inference, please modify the image path and weight path in `result.py`.

Example:

```text
weights/best.pt
```

The detection results will be saved according to the output settings in `result.py`.

## 8. File Description

```text
train.py    Training script.
test.py     Evaluation script for testing the model.
result.py   Inference script for one or several images.
datasets/   Dataset directory.
weights/    Directory for pretrained or trained model weights.
```

## 9. Notes

1. The environment follows the official YOLOv10 implementation.
2. The additional required package is `pytorch_wavelets`.
3. Please place the datasets under the `datasets/` directory.
4. Please download `best.pt` from the provided Baidu Netdisk link before testing or inference.
5. If the dataset path is changed, please update the corresponding configuration file or script.

## 10. Citation

If this code is useful for your research, please cite our paper:

```bibtex
@article{qi2026fwdet,
  title={Frequency-Consistent Wavelet Learning for Detail-Preserving Aerial Small Object Detection},
  author={Qi, Zhongmiao and Jiang, Yan and Tao, Jianwen and Zhang, Bolin},
  journal={The Visual Computer},
  year={2026}
}
```
```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
