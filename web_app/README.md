# Web frontend (minimal)

This is a small Flask-based web UI for uploading an image and running inference with the saved model at `models/best.pt`.

Quick start (from project root):

```bash
# activate project venv and run
source .venv_mac/bin/activate
python web_app/app.py

# then open http://127.0.0.1:5000 in your browser
```

Or use the helper script (makes sure the venv is activated):

```bash
./web_app/run_web.sh
```

Notes:
- The app expects `models/best.pt` to exist. We already copied it to `models/` after training.
- For production you'd want to run behind a WSGI server (gunicorn/uvicorn) and add security checks.
