## Frequency-Consistent Wavelet Representation for Lightweight Small Object Detection in UAV

## Abstract
Small object detection in UAV imagery remains challenging due to limited target resolution, dense spatial distributions, and complex background interference. Existing detectors mainly rely on spatial-domain feature extraction and multi-scale fusion, which may progressively weaken the high-frequency details that are critical for distinguishing small objects. To address this issue, we propose FWDet, a lightweight frequency-consistent wavelet representation framework for aerial small object detection. Specifically, a multi-head wavelet-based frequency decomposition module is introduced to explicitly separate low-frequency semantic structures from high-frequency detail cues during hierarchical feature extraction. To further preserve the correspondence between semantic information and fine-grained local details across scales, we design a direction-aware frequency reconstruction module and a frequency-consistent multi-scale fusion strategy, which reconstruct spatial representations from structured frequency components before cross-scale interaction. In this way, FWDet forms a unified decomposition reconstruction fusion pipeline for detail-preserving feature learning. Extensive experiments on the VisDrone2019 and UAVDT benchmarks demonstrate the effectiveness of the proposed method. FWDet achieves 42.5\% mAP@0.5 on VisDrone2019, and 34.9\% mAP@0.5 and 23.1\% mAP@[0.5:0.95] on UAVDT with only 1.53M parameters, showing a favorable balance between detection accuracy and model compactness. Ablation studies and feature response visualizations further verify the contribution of each proposed component.

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
