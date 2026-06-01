import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os


# ==============================
# 🔥 可视化函数（你写的升级版）
# ==============================
def save_activation_overlay(
    feat: torch.Tensor,
    img_bgr: np.ndarray,
    save_path: str,
    alpha: float = 0.5
):
    """
    Convert feature map to heatmap and overlay on original image.
    """

    if feat is None:
        print(f"[WARN] feature is None: {save_path}")
        return

    # [B,C,H,W] → [C,H,W]
    if feat.dim() == 4:
        feat = feat[0]

    # 通道平均
    mean_map = feat.detach().float().mean(dim=0).cpu().numpy()

    # 归一化
    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
    mean_map = np.uint8(255 * mean_map)

    # resize 到原图尺寸（关键）
    heatmap = cv2.resize(mean_map, (img_bgr.shape[1], img_bgr.shape[0]))

    # 可选：平滑（论文更好看）
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

    # 伪彩色
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # overlay
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    # 保存
    cv2.imwrite(save_path, overlay)


# ==============================
# 🔥 主流程
# ==============================
def main():
    weights = "runs/detect/high_freq_only/weights/best.pt"  # 你的模型
    img_path = "input.jpg"                          # 测试图片
    save_dir = "vis_results2"

    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    yolo = YOLO(weights)
    model = yolo.model.to(device).eval()

    # 读取原图（用于overlay）
    img_bgr = cv2.imread(img_path)

    # 模型输入（640×640）
    img_resized = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    x = torch.from_numpy(x).to(device)

    # ==============================
    # 🔥 注册hook（选择你想看的层）
    # ==============================
    activations = {}

    def get_hook(name):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                activations[name] = out[0]
            else:
                activations[name] = out
        return hook

    # 👉 这里填你要看的层（根据你的结构）
    target_layers = {
        "decomp": 14,    # SplitFreq
        "recon": 17,     # UPFusion
        "fusion": 20    # 后续融合层
    }

    handles = []
    for name, idx in target_layers.items():
        print(f"Hooking layer {idx} → {name}")
        handles.append(
            model.model[idx].register_forward_hook(get_hook(name))
        )

    # ==============================
    # 🔥 前向传播
    # ==============================
    with torch.no_grad():
        _ = model(x)

    # ==============================
    # 🔥 保存可视化
    # ==============================
    for name, feat in activations.items():
        save_path = os.path.join(save_dir, f"{name}.jpg")
        save_activation_overlay(feat, img_bgr, save_path)
        print(f"[Saved] {save_path}")

    # 保存原图
    cv2.imwrite(os.path.join(save_dir, "input.jpg"), img_bgr)

    # 清理hook
    for h in handles:
        h.remove()

    print("✅ Done!")


if __name__ == "__main__":
    main()