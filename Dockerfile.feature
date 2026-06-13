FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    g++ \
    ninja-build \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torchac \
    torchvision \
    numpy \
    pillow

RUN python3 -c "import torchac; print('torchac ready')"

WORKDIR /app

COPY src/ src/
COPY plans/mi300x-training-sprint/fpn_backbone.py fpn_backbone.py
COPY pyproject.toml .
COPY README.md .

RUN pip install -e .

# Checkpoint mounted at runtime
ENV SENTINEL_MODEL=/app/model.pt

ENTRYPOINT ["python3", "-m", "lewm_vc.feature_codec"]
CMD ["--help"]
