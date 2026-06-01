from ultralytics import YOLO

# 关键：加载训练好的权重文件（替换为你实际的权重路径）
# 训练后的权重默认保存在 runs/detect/train/weights/ 下
model = YOLO("runs/detect/UAVDT_high_freq_and_low_freq_fusion_in_FPN/weights/best.pt")  # 核心修改：用 .pt 而非 .yaml

# 执行验证（此时模型输出是正常张量，不会报 dict 错误）
metrics = model.val(
    data='datasets/UAVDT/UAVDT.yaml',
    split='val',
    device=0,
    imgsz=640,
    verbose=True,
    conf=0.4,  # 降低置信度阈值，适配双分支更多检测框
    iou=0.5    # 降低 NMS IoU 阈值，避免漏检
)

# 打印验证指标
print(f"Test 集 mAP@0.5: {metrics.box.map50:.4f}")
print(f"Test 集 mAP@0.5:0.95: {metrics.box.map:.4f}")