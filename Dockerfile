FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (volumes override this in dev)
COPY . .

# Create directories for data and checkpoints
RUN mkdir -p data checkpoints

CMD ["python", "train.py"]
