from ultralytics import YOLO


model = YOLO('yolov8n.pt') # small head is enough
model.train(
data='training/yolo/data_lights.yaml',
epochs=50,
imgsz=640,
batch=32,
device=0,
lr0=0.003,
mosaic=0.5,
mixup=0.0,
workers=8
)
model.val()
model.export(format='onnx', opset=12, dynamic=True)