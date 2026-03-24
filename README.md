# Smart Traffic

A computer vision application for detecting vehicles and traffic lights using YOLOv8.

## Quick Start (Docker)

1.  **Build and Run**:
    ```bash
    docker compose up --build
    ```

2.  **Access the Web App**:
    Open [http://localhost:5000](http://localhost:5000) in your browser.

3.  **Usage**:
    - Upload an image to run inference.
    - Results will be displayed with bounding boxes.

## Development (Local)

If you prefer to run locally without Docker:

```bash
# 1. Create/Activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r infra/requirements.txt  # Note: Check specific web_app needs if this fails

# 3. Run Web App
python web_app/app.py
```

## Structure
- `web_app/`: Flask application.
- `training/`: Model training scripts.
- `models/`: Trained models (`best.pt`).
- `config/`: Configuration files.
