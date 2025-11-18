import cv2
import json
from ultralytics import YOLO
from workers.rules.red_light import RedLightRule
from workers.tracking.bytetrack_wrapper import ByteTrackWrapper
from workers.evidence.builder import EvidenceBuilder

CFG = json.load(open('configs/camera_sample.json'))

veh_model = YOLO('runs/detect/train/weights/best.pt')      # vehicle detection model
light_model = YOLO('runs/detect/train_lights/weights/best.pt')  # traffic-light model

rule = RedLightRule(CFG)
tracker = ByteTrackWrapper()
be = EvidenceBuilder()

cap = cv2.VideoCapture('sample.mp4')

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Vehicle detection
    veh = veh_model(frame, conf=0.35, imgsz=960)[0]
    dets = []
    for b in veh.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        dets.append((x1, y1, x2, y2, cls, conf))

    # Light detection
    lights = light_model(frame, conf=0.40, imgsz=640)[0]
    state = 'G'
    for b in lights.boxes:
        cls = int(b.cls[0])
        state = ['red', 'yellow', 'green'][cls][0].upper()  # R/Y/G
        break

    rule.update_light(state)

    # Tracking
    t = cv2.getTickCount() / cv2.getTickFrequency()
    tracks = tracker.update(dets, t)

    # Evaluate violations
    events = rule.evaluate(tracks)

    for ev in events:
        x1, y1, x2, y2 = map(int, ev['bbox'])
        crop = frame[max(0,y1):y2, max(0,x1):x2].copy()
        meta = {'type': ev['type'], 'time': ev['time']}
        path = be.save_packet(frame, crop, ev['state'], meta)
        print("Saved violation evidence at:", path)

    # Optional visualization
    # cv2.imshow("preview", frame)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
