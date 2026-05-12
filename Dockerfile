FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir flask requests opencv-python-headless pillow numpy

COPY src/ ./src/
COPY checkpoint/ ./checkpoint/

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "-m", "src.server"]