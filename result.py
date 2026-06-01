import os
from pathlib import Path
from ultralytics import YOLO


def run_predict(
    weights: str,
    source: str,
    save_dir: str = "runs/predict_fwdet",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0"
):
    """
    Run inference using a trained Ultralytics YOLO model.

    Args:
        weights: Path to trained model weights, e.g. best.pt
        source: Image path, folder path, video path, or camera index
        save_dir: Directory to save prediction results
        imgsz: Inference image size
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: CUDA device id, e.g. "0", or "cpu"
    """
    model = YOLO(weights)

    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        save_txt=False,      # set True if you want txt labels
        save_conf=True,      # save confidence in txt when save_txt=True
        show=False,
        project=save_dir,
        name="exp",
        exist_ok=True,
        line_width=2,
        boxes=True,
    )

    print(f"Prediction finished. Results saved to: {Path(save_dir) / 'exp'}")
    return results


if __name__ == "__main__":
    weights = r"runs/detect/high_freq_and_low_freq_fusion_in_FPN/weights/best.pt"   # 改成你的权重路径
    source = r"test_img"                          # 可以是单张图片，也可以是文件夹
    run_predict(
        weights=weights,
        source=source,
        save_dir="predict_fw",
        imgsz=640,
        conf=0.4,
        iou=0.5,
        device="0"
    )