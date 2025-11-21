from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from pathlib import Path
import uuid
import os
import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / 'web_app' / 'static'
UPLOAD_DIR = STATIC_DIR / 'uploads'
RESULT_DIR = STATIC_DIR / 'results'

for d in (UPLOAD_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(APP_ROOT / 'web_app' / 'templates'))

# Load model at startup. Allow overriding via the MODEL_PATH env var.
# If MODEL_PATH is not set, default to the working pretrained model 'yolov8n.pt'
# which provides multi-class vehicle detections out of the box.
MODEL_PATH = os.environ.get('MODEL_PATH', None)
if MODEL_PATH is None:
    # prefer a local copy in models/ if present, else fall back to the ultralytics
    # pretrained name 'yolov8n.pt' (will be downloaded/used by ultralytics if needed)
    local_pref = APP_ROOT / 'models' / 'yolov8n.pt'
    if local_pref.exists():
        MODEL_PATH = str(local_pref)
    else:
        MODEL_PATH = 'yolov8n.pt'
else:
    # allow environment var to be either a relative/local path or a full path
    MODEL_PATH = str(MODEL_PATH)
# Target dataset class names (the app will map model output names to these)
TARGET_NAMES = ['car', 'bus', 'truck', 'motorbike', 'person']

# model -> target mapping will be computed at startup so we can translate
# pretrained-model class ids (COCO) into our dataset ids (0..len(TARGET_NAMES)-1)
model = None
model_to_target_map = {}  # maps model_cls_id -> target_cls_id (or None)
try:
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print('Loaded model:', MODEL_PATH)
    # build mapping from model names to our target ids
    if hasattr(model, 'names') and isinstance(model.names, dict):
        for mid, mname in model.names.items():
            if isinstance(mname, bytes):
                mname = mname.decode('utf-8')
            lname = str(mname).lower()
            if lname in TARGET_NAMES:
                model_to_target_map[int(mid)] = TARGET_NAMES.index(lname)
            else:
                # leave unmapped classes as None (they won't be counted)
                model_to_target_map[int(mid)] = None
        print('Model->target mapping:', model_to_target_map)
except Exception as e:
    print('Failed to load model at startup:', MODEL_PATH, e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    uid = uuid.uuid4().hex
    filename = f'{uid}_{file.filename}'
    upload_path = UPLOAD_DIR / filename
    file.save(upload_path)

    if model is None:
        return 'Model not loaded on server. Check server logs.', 500

    # Run prediction in-memory and save outputs ourselves. Avoid Ultralytics' auto-save
    # to prevent duplicate annotations — we'll draw a single overlay using the
    # mapped target names and write remapped YOLO .txt labels.
    name = f'run_{uid}'
    out_dir = RESULT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = out_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.predict(source=str(upload_path), conf=0.25, save=False)
    except Exception:
        # fallback single-call
        results = model(str(upload_path), conf=0.25)

    img_rel = None
    try:
        res = results[0]
        b = getattr(res, 'boxes', None)
        boxes_xyxy = None
        confs = None
        cls_ids = None
        if b is not None:
            try:
                boxes_xyxy = b.xyxy.cpu().numpy()
            except Exception:
                boxes_xyxy = np.array(b.xyxy) if hasattr(b, 'xyxy') else None
            try:
                confs = b.conf.cpu().numpy()
            except Exception:
                confs = np.array(b.conf) if hasattr(b, 'conf') else None
            try:
                cls_ids = b.cls.cpu().numpy()
            except Exception:
                cls_ids = np.array(b.cls) if hasattr(b, 'cls') else None

        img = cv2.imread(str(upload_path))
        h, w = img.shape[:2]
        saved_img_path = out_dir / (f"{uid}_{upload_path.name}")
        yolo_lines = []
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            for i, xy in enumerate(boxes_xyxy):
                try:
                    x1, y1, x2, y2 = map(int, xy[:4])
                except Exception:
                    continue
                cid = int(cls_ids[i]) if cls_ids is not None else None
                conf = float(confs[i]) if confs is not None else None
                mapped = None
                if cid is not None:
                    mapped = model_to_target_map.get(int(cid)) if model_to_target_map else None
                if mapped is None:
                    continue
                cname = TARGET_NAMES[mapped] if mapped < len(TARGET_NAMES) else str(mapped)
                text = f"{cname} {conf:.2f}" if conf is not None else f"{cname}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                ty = max(0, y1 - th - baseline - 4)
                cv2.rectangle(img, (x1, ty), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                yolo_lines.append(f"{mapped} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        cv2.imwrite(str(saved_img_path), img)
        img_rel = os.path.relpath(str(saved_img_path), STATIC_DIR)
        if yolo_lines:
            label_file = label_dir / (saved_img_path.stem + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')
    except Exception:
        try:
            saved = out_dir / upload_path.name
            upload_path.replace(saved)
            img_rel = os.path.relpath(str(saved), STATIC_DIR)
        except Exception:
            img_rel = None

    if img_rel:
        return redirect(url_for('index', result=img_rel))

    # fallback: if no image saved, render index with an error
    return render_template('index.html', img_url=None, class_counts={},)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve project static files rooted at `web_app/static`.

    We use a custom route so that `/static/<path>` serves files from
    the `STATIC_DIR` directory. The template constructs URLs like
    `/static/results/run_xxx/imagename.jpg` where the path after
    `/static/` is relative to `STATIC_DIR`.
    """
    return send_from_directory(str(STATIC_DIR), filename)


@app.route('/_result_counts')
def result_counts():
    """Return JSON counts for a given result image path (img_url prefixed with /static/...).
    Query param: path=/static/....jpg
    """
    from flask import jsonify, request
    p = request.args.get('path')
    if not p:
        return jsonify({})

    # p is expected to be something like '/static/results/run_xxx/imagename.jpg'
    # remove the leading '/static/' and compute the labels directory under STATIC_DIR
    if p.startswith('/static/'):
        rel = p[len('/static/'):]
    else:
        rel = p
    rel_path = Path(rel)
    out_dir = STATIC_DIR / rel_path.parent
    label_dir = out_dir / 'labels'
    class_counts = {}
    if label_dir.exists():
        for txt in label_dir.glob('*.txt'):
            with open(txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except Exception:
                        continue
                    cname = TARGET_NAMES[cid] if cid < len(TARGET_NAMES) else str(cid)
                    class_counts[cname] = class_counts.get(cname, 0) + 1
    return jsonify(class_counts)

if __name__ == '__main__':
    # Run with the project's venv activated for correct env
    app.run(host='127.0.0.1', port=5000, debug=False)
