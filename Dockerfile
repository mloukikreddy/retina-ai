FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV + TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create uploads folder
RUN mkdir -p static/uploads

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["python", "app.py"]