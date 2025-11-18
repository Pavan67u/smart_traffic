from ultralytics import YOLO


model = YOLO('yolov8n.pt')  # nano model instead of small
model.train(
data='training/yolo/data_vehicles.yaml',
epochs=10,  # reduced epochs for faster testing
imgsz=640,  # reduced image size for nano model
batch=8,
device='cpu',
lr0=0.003,
optimizer='adamw',
mosaic=1.0,
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
workers=8
)
model.val()
model.export(format='onnx', opset=12, dynamic=True)
model.predict(source='training/yolo/test_images', conf=0.25, save=True, save_txt=True)