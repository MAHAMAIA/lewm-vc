# Building the Sentinel Edge Gateway

## Overview

The Sentinel Edge Gateway is a Jetson Orin-based appliance that runs the Sentinel codec as a Docker container. It ingests RTSP camera feeds, compresses them using the LeWM-VC codec, and serves both base layer (semantic latents for cloud AI) and enhancement layer (local storage for forensic reconstruction).

---

## Hardware BOM

| Component | Model | Qty | Approx Cost |
|-----------|-------|-----|-------------|
| Module | Jetson Orin NX 16GB | 1 | $600 |
| Carrier Board | Seeed A603 or Waveshare Orin carrier | 1 | $150-250 |
| Storage | NVMe SSD 256GB (Samsung 980 or similar) | 1 | $50 |
| Power Supply | 12V/5A DC adapter (barrel jack) | 1 | $20 |
| Cooling | Active cooler for Orin NX | 1 | $15 |
| Enclosure | IP40 metal enclosure, 150x110x50mm | 1 | $30 |
| Optional | PoE+ injector/hat for single-cable power+data | 1 | $40 |

**Total BOM (1 unit): ~$900-1,000**

### Important: Orin NX Required (Not Nano)

Jetson Orin Nano lacks NVENC hardware encoding. Software H.265 at 1080p on Nano's ARM cores achieves only 3-8 fps — insufficient for real-time multi-stream. **Use Orin NX 16GB or AGX only.**

### Pre-built Options

| Model | Pros | Cons | Price |
|-------|------|------|-------|
| NVIDIA Jetson Orin NX Developer Kit | Official, well-supported | No enclosure, developer form factor | $599 (module+carrier) |
| Auvidea JNX42-NX | Rugged, industrial temp range | More expensive | $850 |
| Seeed reComputer J4012 | Complete with enclosure, RTC, PoE | Limited availability | $750 |

### Scale Pricing (Estimate)

| Volume | Unit Cost | Notes |
|--------|-----------|-------|
| 1-10 | $900-1,000 | Retail modules + carrier boards, hand assembly |
| 10-50 | $800-900 | Distributor pricing on modules, batch carrier order |
| 50-100+ | $700-800 | ODM pricing on module + custom carrier integration |

---

## Software Stack

```
┌─────────────────────────────────────────────┐
│  RTSP Camera Feeds                           │
│  (up to 4 streams, 1080p @ 30fps)           │
├─────────────────────────────────────────────┤
│  Sentinel Docker Container                   │
│  ┌───────────────────────────────────────┐   │
│  │  sentinel-encode (Python)             │   │
│  │  ├── RTSP ingestion per stream        │   │
│  │  ├── LeWM-VC encoder                  │   │
│  │  ├── Base layer → cloud API (MQTT)    │   │
│  │  └── Enhancement layer → local NVMe   │   │
│  ├───────────────────────────────────────┤   │
│  │  sentinel-api (Flask, port 5000)      │   │
│  │  ├── /health                          │   │
│  │  ├── /status (stream stats)           │   │
│  │  └── /reconstruct (fetch EL + decode) │   │
│  └───────────────────────────────────────┘   │
├─────────────────────────────────────────────┤
│  Docker (GPU passthrough via nvidia-ctk)     │
├─────────────────────────────────────────────┤
│  JetPack 6.0 (CUDA 12.2, cuDNN, TensorRT)   │
├─────────────────────────────────────────────┤
│  NVIDIA Jetson Orin NX 16GB                  │
└─────────────────────────────────────────────┘
```

---

## Step 1: Flash Jetson Orin

### Option A: SDK Manager (recommended, requires x86 host)
```bash
# On an x86 Ubuntu machine, install SDK Manager
# Download JetPack 6.0 from NVIDIA
# Select Jetson Orin NX 16GB target
# Flash with:
#   - Ubuntu 22.04
#   - CUDA 12.2
#   - cuDNN 8.9
#   - TensorRT 8.6
```

### Option B: SD Card Image (no host PC needed)
```
# Download the pre-built JetPack 6.0 SD card image for Orin NX
# Flash to an NVMe SSD (not SD card — too slow):
sudo dd if=jetpack-6.0-jp6.0-tegra-aarch64.img of=/dev/nvme0n1 bs=4M status=progress
```

### Initial Setup
```bash
# Boot the Orin, run first-time setup
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv docker.io nvidia-jetpack
```

---

## Step 2: Install NVIDIA Container Toolkit

```bash
# Required for GPU access inside Docker
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
sudo docker run --rm --runtime=nvidia nvidia/cuda:12.2-runtime nvidia-smi
```

---

## Step 3: Build the Sentinel Docker Image

### Create a Jetson-optimized Dockerfile

This is the same as `lewmvc-stream/Dockerfile` but uses a CUDA base image and installs PyTorch for aarch64:

```dockerfile
# sentinel-gateway.dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.3.0_pth2.1.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir flask requests opencv-python-headless pillow numpy paho-mqtt

COPY src/ ./src/
COPY checkpoint/ ./checkpoint/

ENV PYTHONUNBUFFERED=1
ENV SENTINEL_DEVICE=cuda

EXPOSE 5000

CMD ["python", "-m", "src.server"]
```

### Build
```bash
# On the Jetson Orin
docker build -f sentinel-gateway.dockerfile -t sentinel-gateway:latest .
```

---

## Step 4: Configure RTSP Camera Inputs

```bash
# Create config directory
mkdir -p /etc/sentinel

# Camera configuration
cat > /etc/sentinel/cameras.yml << 'EOF'
cameras:
  cam01:
    uri: rtsp://192.168.1.100:554/stream1
    fps: 30
    resolution: 1920x1080
    enable_bl: true
    enable_el: true
  cam02:
    uri: rtsp://192.168.1.101:554/stream1
    fps: 15
    resolution: 1280x720
    enable_bl: true
    enable_el: false  # low-priority camera, base layer only

mqtt:
  broker: cloud.mahamaia.com
  port: 1883
  topic_prefix: sentinel/gateway/{gateway_id}

storage:
  el_path: /mnt/nvme/enhancement_layer
  retention_days: 90
EOF
```

---

## Step 5: Run as System Service

```bash
# Create systemd service
cat > /etc/systemd/system/sentinel-gateway.service << 'EOF'
[Unit]
Description=Sentinel Edge Gateway
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker stop sentinel-gateway
ExecStartPre=-/usr/bin/docker rm sentinel-gateway
ExecStart=/usr/bin/docker run \
    --rm \
    --name sentinel-gateway \
    --runtime=nvidia \
    --gpus all \
    --network host \
    -v /etc/sentinel:/app/config:ro \
    -v /mnt/nvme/enhancement_layer:/app/data/enhancement \
    -e SENTINEL_CONFIG=/app/config/cameras.yml \
    -e SENTINEL_GATEWAY_ID=$(cat /etc/sentinel/gateway_id) \
    sentinel-gateway:latest

ExecStop=/usr/bin/docker stop sentinel-gateway

[Install]
WantedBy=multi-user.target
EOF

# Set gateway identity
echo "gw-$(hostname -I | awk '{print $1}' | tr '.' '-')" > /etc/sentinel/gateway_id

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable sentinel-gateway
sudo systemctl start sentinel-gateway
```

---

## Step 6: Verify

```bash
# Check service status
systemctl status sentinel-gateway

# Check API health
curl http://localhost:5000/health

# Check stream status
curl http://localhost:5000/status

# View logs
journalctl -u sentinel-gateway -f
```

---

## Step 7: Network Configuration

### Satellite/VSAT Link (Production)
```ini
# /etc/sentinel/network.conf
[link]
type = vsat
max_uplink = 5   # Mbps
max_downlink = 20 # Mbps
latency_ms = 600

[quality]
base_layer_qp = 28
enhancement_qp = 22
target_bpp = 0.15
```

### Cloud Relay (MQTT + HTTP)
The base layer latents are small (~0.003 BPP per frame) and can be transmitted over VSAT continuously. The enhancement layer stays local and is fetched via HTTP on demand.

---

## Production Checklist

- [ ] Jetson Orin flashed with JetPack 6.0
- [ ] NVIDIA Container Toolkit installed and verified
- [ ] Docker image built with GPU passthrough
- [ ] Camera config YAML written
- [ ] Systemd service installed and running
- [ ] MQTT broker credentials configured
- [ ] NVMe storage mounted at `/mnt/nvme`
- [ ] Firewall: only ports 5000 (internal) and outbound MQTT (8883)
- [ ] Watchdog timer enabled (`sudo apt install watchdog`)
- [ ] Temperature monitoring (`tegrastats`)

---

## Cloud Monitoring API

The gateway exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Heartbeat + uptime |
| `/status` | GET | Per-stream encode stats (fps, bpp, dropped frames) |
| `/reconstruct` | POST | Request full frame reconstruction from EL + BL |
| `/logs` | GET | Recent log lines |
| `/config` | GET | Current runtime configuration |

---

## Building at Scale

For quantities >10 units, consider a pre-built solution:

1. **ODM partner** — Integrate with a carrier board manufacturer (e.g., Seeed, Auvidea) for custom enclosure + thermal design
2. **Pre-flash** — Have the ODM flash JetPack and pre-load the Sentinel Docker image
3. **Ship config** — Gateway auto-configures on first boot via MQTT (download camera config from cloud)
4. **Zero-touch provisioning** — Generate per-unit credentials at manufacturing time

Target unit cost at Q100: **~$700-800** (down from $900-1,000 at Q1), dependent on ODM pricing for Orin NX module + custom carrier integration.

---

## Legal & Operational Notes for Deployment

### Legal Admissibility of Neural Reconstruction
- Purely generative reconstruction of video from abstract semantic latents faces legal admissibility challenges under *State v. Rittenhouse* (2021), where AI-enhanced video was excluded due to opaque neural interpolation methods.
- However, Sentinel's dual-layer SVC stores a **deterministic residual enhancement layer** on local NVMe. When fused with the base layer, this produces a mathematically invertible reconstruction — not a generative "hallucination." No court has ruled on such a system.
- **Recommendation:** For chain-of-custody claims, keep the base layer bitstream (continuous) + enhancement layer (local) as primary evidence. The full fused reconstruction may be admissible in civil/industrial use cases (lower burden of proof). Criminal admissibility requires a test case.

### Starlink Throttling Advantage
- Starlink imposes a 512 kbps uplink throttle after data cap is exceeded on some plans.
- Sentinel's base layer (target <0.015 bpp at 1080p) can still stream continuously under throttle, while H.265 at equivalent quality fails.
- This is a key selling point: "Your cameras keep sending usable AI data even when Starlink throttles the connection."
- Benchmark this scenario in the pilot: measure BL-only frame rate under simulated Starlink throttle conditions.

### Legal Consultation Preparation
- Document the deterministic residual enhancement layer as a formal specification (e.g., a whitepaper) for legal counsel review.
- The specification should demonstrate that EL + BL fusion is a mathematically invertible process — not a generative or opaque neural interpolation — to distinguish from *State v. Rittenhouse* precedent.
- Recommended deliverable: a one-page technical brief describing the SVC architecture, the deterministic fusion path, and the mathematical relationship between input video and reconstructed 192-ch latent.
