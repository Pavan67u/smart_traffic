# Indian Road & Vehicle Datasets for Fine-Tuning

This guide lists **publicly available datasets** for Indian roads, vehicles, and traffic conditions that you can use to fine-tune the Smart Traffic model. Indian traffic differs from Western benchmarks: mixed vehicle types (two-wheelers, auto-rickshaws, carts), unstructured lanes, and varied lighting/weather.

---

## Quick picks (easiest to get started)

| Dataset | Size | Classes | Format | Access |
|--------|------|---------|--------|--------|
| **Roboflow Indian Roads** | ~1k images | 11 (car, bus, truck, auto, two-wheeler, person, animal, cart, rickshaw, etc.) | YOLO | Free download from Roboflow; export as YOLOv8 |
| **UVH-26** (IISc) | 26.6k images | 14 India-specific (2W, auto-rickshaw, LCV, bus, etc.) | COCO / convert to YOLO | HuggingFace (CC-BY-4.0) |
| **IRUVD** | 4k images, 14.3k boxes | 14 vehicle types + pedestrian | Check repo | GitHub + Google Drive |

---

## 1. Roboflow – YOLO V8 Indian Roads Dataset

- **Link:** [Roboflow Universe – YOLO V8 Indian roads dataset](https://universe.roboflow.com/object-detection-dp5wa/yolo-v8-indian-roads-dataset)
- **Size:** ~1,000 images  
- **Classes:** Animal, Auto (auto-rickshaw), Bus, Car, Carts, Person, Rickshaw, Truck, **Two-wheeler** (and variants)
- **Format:** YOLO (export as “YOLOv8” from Roboflow)
- **License:** CC BY 4.0  
- **Why use:** No approval step; create a free Roboflow account → open project → **Download** → choose **YOLOv8** → download ZIP. Unzip into a folder and point `data_vehicles.yaml` at it (see “Using a downloaded dataset” below).

---

## 2. UVH-26 (IISc – Bengaluru Safe City)

- **Link:** [HuggingFace – iisc-aim/UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26)
- **Size:** 26,646 images, ~1.8M bounding boxes  
- **Classes:** 14 India-specific (e.g. two-wheeler, auto-rickshaw, light commercial vehicle, bus, car, truck, etc.)
- **Format:** Check dataset card for COCO/YOLO; if COCO, convert to YOLO (script below or Roboflow).
- **License:** CC-BY-4.0  
- **Why use:** Large, real CCTV-style urban Indian traffic; includes pre-trained models (YOLOv11, RT-DETR, etc.) you can compare against or use as teacher.

**Download example (HuggingFace):**
```bash
pip install huggingface_hub datasets
# Then in Python or use the HuggingFace CLI to download the dataset.
```

---

## 3. DriveIndia (TiHAN – IIT Hyderabad)

- **Link:** [TiHAN TiAND Datasets](https://tihan.iith.ac.in/tiand-datasets/)  
- **Paper:** [DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes (arXiv)](https://arxiv.org/html/2507.19912v3)
- **Size:** 66,986 images (1920×1080), 24 categories  
- **Conditions:** Urban, rural, highway; fog, rain, low light; dense mixed traffic  
- **Format:** YOLO (normalized .txt per image)  
- **Access:** Request via [TiHAN form](https://docs.google.com/forms/d/e/1FAIpQLSfW32O35CgZU0TSv-m2tiKtKs9SxZYNFpz2mrZlK4FHzTNC2A/viewform); download link shared after approval.

---

## 4. IRUVD (Indian Road User Vehicle Dataset)

- **Link:** [GitHub – IRUVD/IRUVD](https://github.com/IRUVD/IRUVD)  
- **Size:** 4,000 images, 14,300 bounding boxes  
- **Classes:** 14 (13 vehicle types + pedestrian); urban and rural (e.g. West Bengal and other cities)  
- **Format:** Check repo for structure; may need a small script to convert to YOLO train/val layout.  
- **Download:** Google Drive link in the GitHub README.

---

## 5. Indian Driving Dataset (IDD)

- **Link:** [IDD – IIIT Hyderabad](https://idd.insaan.iiit.ac.in/)  
- **Size:** ~46k images (train/val/test splits), 10k+ for detection  
- **Conditions:** Unstructured roads, varied vehicles (e.g. auto-rickshaws, animals)  
- **Format:** Check official site for detection annotations (COCO or custom).  
- **Access:** Register / request on the official portal.

---

## 6. NanoNets Indian Roads Object Detection

- **Link:** [GitHub – NanoNets/IndianRoadsObjectDetectionDataset](https://github.com/NanoNets/IndianRoadsObjectDetectionDataset)  
- **Content:** Object detection for Indian roads; structure and format described in the repo.  
- **Use:** Clone repo and follow README for download and folder layout.

---

## Using a downloaded dataset in this project

### Option A – Dataset already in YOLO layout

If you have:

```
dataset_root/
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/   (optional)
  test/labels/
```

1. Copy or symlink `dataset_root` to `data/vehicles_yolo` (or any path you prefer).  
2. Edit **`training/yolo/data_vehicles.yaml`**:

```yaml
path: data/vehicles_yolo   # or your path relative to repo root
train: train/images
val: val/images
test: test/images
names: [car, bus, truck, motorbike, person]   # must match your label class indices
```

3. **Class names and indices:** The `names` list must match the **integer class IDs** in your `.txt` labels (0 = first class, 1 = second, etc.). If the Indian dataset uses different names (e.g. `auto_rickshaw`, `two_wheeler`), set `names` to that exact order so index 0 is the first class in the dataset, and so on.  
4. Run finetune:

```bash
./scripts/run_finetune.sh --data training/yolo/data_vehicles.yaml --epochs 30 --device mps
```

### Option B – Roboflow export

1. Open the [Roboflow Indian roads project](https://universe.roboflow.com/object-detection-dp5wa/yolo-v8-indian-roads-dataset).  
2. Click **Download** → choose **YOLOv8** → download.  
3. Unzip into e.g. `data/indian_roads_roboflow`.  
4. Ensure the folder has `train/images`, `train/labels`, `val/images`, `val/labels`. (Roboflow sometimes uses `valid` instead of `val`; if so, rename `valid` → `val` or set `val: valid/images` in the YAML.)  
5. Create **`training/yolo/data_indian_roads.yaml`**:

```yaml
path: data/indian_roads_roboflow
train: train/images
val: val/images
test: test/images
names: [Animal, Auto, Bus, Car, Carts, Person, Rikshaw, Truck, Two-wheeler, ...]  # copy exact class names from Roboflow
```

6. Train:

```bash
./scripts/run_finetune.sh --data training/yolo/data_indian_roads.yaml --epochs 30
```

### Option C – Mapping Indian classes to the app’s 5 classes

The web app uses: **car, bus, truck, motorbike, person**. If your Indian dataset has more classes (e.g. auto-rickshaw, two-wheeler, rickshaw), you can:

- **Train on all Indian classes** and keep the model’s native class names in the app (update `TARGET_NAMES` in `web_app/app.py` and any mapping to match the new `names` in the YAML), **or**
- **Map** Indian classes into these 5: e.g. map “Auto”, “Rikshaw” → one class; “Two-wheeler”, “motorbike” → motorbike; “Car” → car; “Bus” → bus; “Truck” → truck; “Person” → person. Mapping can be done by a small script that rewrites the `.txt` files and assigns new class IDs (0–4) so that `names: [car, bus, truck, motorbike, person]` is correct.

If you want, a small **class-mapping script** can be added under `training/utils/` to convert any YOLO dataset to a fixed 5-class scheme for this app.

---

## Summary

| Goal | Suggested dataset |
|------|--------------------|
| Fastest start, no approval | **Roboflow Indian Roads** (download → YOLOv8 → train) |
| Large, CCTV-style, open license | **UVH-26** (HuggingFace) |
| Maximum scale, diverse conditions | **DriveIndia** (request via TiHAN form) |
| Medium size, still images | **IRUVD** (GitHub + Drive) or **IDD** (IIIT portal) |

For **Indian roads and conditions**, prefer datasets that include **two-wheelers**, **auto-rickshaws**, and **mixed traffic**; they will transfer better to real Indian deployments than generic COCO-only training.
