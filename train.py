from ultralytics import YOLO
from ultralytics import settings
settings.update({"wandb": False})
model = YOLO("ultralytics/cfg/models/v10/yolov10n.yaml")

model.train(
    data="datasets/Visdrone2019/A.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    
    amp=False,
    # wandb=False
)
