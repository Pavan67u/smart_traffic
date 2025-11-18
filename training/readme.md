Goal: two models
(1) vehicles: car/bus/truck/motorbike/person
(2) traffic_lights: red/yellow/green


Steps:
1) Collect videos similar to your cameras. Extract frames.
2) Label with labelimg/roboflow in YOLO format.
3) Split into train/val/test using split_dataset.py.
4) Edit data yaml files to point to your paths.
5) Train vehicles, then lights.