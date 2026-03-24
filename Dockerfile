# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Explicitly listing dependencies to avoid relying on the potentially mismatching infra/requirements.txt
RUN pip install --no-cache-dir \
    flask \
    flask-sqlalchemy \
    ultralytics \
    opencv-python-headless \
    lapx \
    scipy \
    numpy \
    pillow

# Copy the entire project
COPY . /app

# Create necessary directories for runtime data
RUN mkdir -p /app/web_app/static/uploads /app/web_app/static/results

# Expose the Flask port
EXPOSE 5050

# Run the application
CMD ["python", "web_app/app.py"]
