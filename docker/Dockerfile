FROM ultralytics/ultralytics:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ensure minimal runtime deps: ca-certificates for pip and libGL/glib for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (Ultralytics image already contains torch/ultralytics/OpenCV in most tags)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /opt/app
COPY .. /opt/app

CMD ["bash"]
