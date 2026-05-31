# Production Deployment Pipeline

## Status

❌ **Not built.** No deployment infrastructure exists. The model trains on
MI300X, checkpoints live inside the Docker container, and there's no
mechanism for remote updates, monitoring, or customer onboarding.

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                  MAHAMAIA Cloud                  │
│                                                  │
│  ┌─────────────┐  ┌──────────────┐              │
│  │ Model Store  │  │ API Server   │              │
│  │ S3/GCS       │  │ /v1/models/  │              │
│  │ best.pt      │  │ /v1/usage/   │              │
│  │ tensorrt.eng  │  │ /v1/status/  │              │
│  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                      │
└─────────┼─────────────────┼──────────────────────┘
          │                 │
          │ HTTPS           │ HTTPS
          │                 │
┌─────────▼─────────────────▼──────────────────────┐
│              Customer Site                        │
│                                                   │
│  ┌───────────────────────────────────────────┐   │
│  │ Jetson Orin NX (Jetson)              │   │
│  │                                           │   │
│  │  ┌─────────┐  Camera Feed (RTSP)          │   │
│  │  │ agent   │◄────── H.264/IP camera       │   │
│  │  │ .py     │                              │   │
│  │  │         │  ┌──────────┐  ┌──────────┐  │   │
│  │  │         │  │ Encoder  │  │ Uploader │  │   │
│  │  │         │  │ TRT eng  │  │ S3 sync  │  │   │
│  │  └─────────┘  └──────────┘  └──────────┘  │   │
│  │                                           │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Agent (Jetson-side Python service)

```
/opt/lewm-vc/
├── agent.py              # Main service loop
├── models/
│   ├── encoder.eng       # TensorRT engine (current)
│   ├── predictor.eng
│   ├── decoder.eng
│   └── entropy.eng
├── config.yaml           # Site-specific config
├── bitstream/            # Encoded output
│   └── 2026-06-01/
│       └── site_cam1_001.lewm
└── logs/
    └── agent.log
```

#### agent.py main loop:

```python
class LeWMAgent:
    """
    Runs on Jetson at customer site.
    Captures camera feed, encodes with LeWM-VC, uploads bitstream to cloud.
    """

    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))
        self.models = self._load_models()
        self.uploader = S3Uploader(self.config["s3"])
        self.encoder = Encoder(self.models)
        self.heartbeat = Heartbeat(self.config["api_server"])

    def run(self):
        camera = Camera(self.config["camera"]["rtsp_url"])
        for frame in camera.stream():
            packet = self.encoder.encode_frame(frame)
            self.uploader.push(packet)
            self.heartbeat.ping({"frames_encoded": 1, "uptime_s": ...})
```

### 2. Model Store & Updates

Models are stored in S3 with versioning:

```
s3://mahamaia-models/
├── production/
│   ├── v1.0/
│   │   ├── encoder.eng       # TensorRT FP16
│   │   ├── predictor.eng
│   │   ├── decoder.eng
│   │   ├── entropy.eng
│   │   └── manifest.json     # Hash, date, metrics
│   └── latest → v1.0
└── staging/
    └── v1.1-beta/
        └── ...
```

Update mechanism:

```python
def check_for_update(current_version: str) -> str | None:
    """Poll API server for newer model version. Download if available."""
    resp = requests.get(f"{API}/v1/models/latest")
    if resp.json()["version"] != current_version:
        download_models(resp.json()["download_url"])
        return resp.json()["version"]
    return None

# Called every hour by agent
version = check_for_update(current_version)
if version:
    restart_service()  # Graceful: finish current GOP, swap engines, continue
```

### 3. API Server

Minimal API for deployment management (could be a single FastAPI instance):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/health` | GET | Server health check |
| `/v1/models/latest` | GET | Latest model version + download URL |
| `/v1/usage` | POST | Agent reports: frames encoded, bytes uploaded |
| `/v1/alerts` | POST | Error reports from agent |
| `/v1/config` | GET | Per-site configuration (lambda, resolution) |

### 4. Bitstream Storage & Delivery

Encoded bitstreams are uploaded to S3 (or customer-provided storage):

```
s3://customer-bucket/
└── lewm-vc/
    └── site_abc123/
        ├── camera_1/
        │   ├── 2026-06-01/
        │   │   ├── 001.lewm
        │   │   ├── 002.lewm
        │   │   └── ...
        │   └── manifest.json
        └── camera_2/
            └── ...
```

### 5. Decoder Service (Cloud-side)

When the customer wants to view footage:

```
Bitstream S3 → Decoder Lambda/ECS → H.264 → CDN → Browser/App
```

The decoder runs the same TensorRT engines in reverse. Since decode is
cheaper than encode (no encoder, no predictor), a single GPU can handle
30+ concurrent streams.

## Installation Script

```bash
#!/bin/bash
# install.sh — Install LeWM-VC agent on Jetson Orin NX

set -e

VERSION=${1:-latest}
API_SERVER=${2:-https://api.mahamaia.ai}

echo "Installing LeWM-VC Agent v$VERSION..."

# Install system deps
apt update
apt install -y python3-pip ffmpeg libavcodec-dev

# Install Python deps
pip3 install torch torchvision numpy opencv-python boto3 requests pyyaml

# Create directories
mkdir -p /opt/lewm-vc/{models,bitstream,logs,config}

# Download model
wget -q "$API_SERVER/v1/models/$VERSION/download" -O /tmp/models.tar.gz
tar xzf /tmp/models.tar.gz -C /opt/lewm-vc/models/

# Install agent
cp agent.py /opt/lewm-vc/
chmod +x /opt/lewm-vc/agent.py

# Create systemd service
cat > /etc/systemd/system/lewm-vc.service << 'SYSTEMD'
[Unit]
Description=LeWM-VC Video Codec Agent
After=network.target

[Service]
ExecStart=/usr/bin/python3 /opt/lewm-vc/agent.py
WorkingDirectory=/opt/lewm-vc
Restart=on-failure
User=lewmvc

[Install]
WantedBy=multi-user.target
SYSTEMD

# Enable service
useradd -r lewmvc
systemctl enable lewm-vc
systemctl start lewm-vc

echo "Installation complete. Status: systemctl status lewm-vc"
```

## Monitoring & Observability

### Agent Heartbeat

Sent every 60 seconds:

```json
{
    "site_id": "abc123",
    "agent_version": "1.0.0",
    "model_version": "v1.0",
    "uptime_seconds": 86400,
    "frames_encoded": 2592000,
    "bytes_uploaded": 1073741824,
    "gpu_util_pct": 72,
    "temperature_c": 68,
    "errors_last_hour": 0,
    "disk_usage_pct": 34
}
```

### Alert Conditions

| Condition | Action |
|-----------|--------|
| GPU utilization drops below 50% for 10 min | Check camera feed / model hang |
| Temperature > 85°C for 5 min | Reduce encode rate / activate fan |
| Upload queue exceeds 1000 packets | Check network / increase batch size |
| Failed to encode 5 consecutive frames | Restart agent, notify admin |

### Dashboard Metrics

Tracked per-site per-camera:

- Frames encoded per hour
- Average BPP (bitrate metric)
- Upload latency (encode → S3)
- GPU utilization (useful signal: encode throughput)
- Error rate

## Security

| Concern | Mitigation |
|---------|-----------|
| Bitstreams contain customer video data | Encrypt at rest (AES-256), in transit (TLS 1.3) |
| Agent fetches model updates | Signed model manifests (Ed25519 signatures) |
| API access token | Per-device tokens, rotated monthly |
| Physical access to Jetson | Disk encryption (LUKS), locked enclosure |

## Cost Model (Pilot)

| Item | Monthly cost | Notes |
|------|-------------|-------|
| Jetson Orin NX (16GB) | $500 (amortized over 12mo) | ~$42/mo |
| AWS S3 storage (1 TB) | $23 | 1 month of 4K footage at 0.5 bpp |
| AWS data transfer (1 TB) | $90 | Satellite uplink cost excluded |
| API server (t2.micro) | $8 | Minimal for pilot |
| **Total per site** | **~$163/mo** | |

For the design pilot, the customer pays hardware + AWS costs (or
MAHAMAIA covers for the first month as proof).

## Pre-Flight Checklist

- [ ] Jetson Orin NX acquired and flashed with JetPack 6.x
- [ ] TensorRT engines built and validated (bit-exact with PyTorch)
- [ ] Python codec module installed and loads models at startup
- [ ] FFmpeg plugin compiles on Jetson
- [ ] Camera RTSP feed captured, encoded, decoded in test
- [ ] S3 upload works with customer bucket permissions
- [ ] API server endpoints respond
- [ ] Agent heartbeat shows in dashboard
- [ ] Manual: flash SD card, run install.sh, verify camera → cloud pipeline
