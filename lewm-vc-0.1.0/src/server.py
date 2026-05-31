"""
LeWM-VC Flask API Server

HTTP server for LeWM-VC video codec inference.
"""

import base64
import io
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, Response

from codec import LeWMVideoCodec, compute_psnr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

codec: LeWMVideoCodec = None
stats = {
    "frames_processed": 0,
    "total_bits": 0,
    "total_time_ms": 0,
    "sessions": 0,
}


def load_codec(checkpoint_path: str = None):
    """Load the codec model."""
    global codec
    if checkpoint_path is None:
        default_paths = [
            "checkpoint/temporal_final.pt",
            "../checkpoint/temporal_final.pt",
        ]
        for path in default_paths:
            if Path(path).exists():
                checkpoint_path = path
                break

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        codec = LeWMVideoCodec(checkpoint_path=checkpoint_path)
    else:
        logger.warning("No checkpoint found, using untrained model")
        codec = LeWMVideoCodec()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": codec is not None,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        }
    )


@app.route("/start", methods=["POST"])
def start_session():
    """Start a new encoding session."""
    global stats
    stats["sessions"] += 1
    codec.reset()
    return jsonify(
        {
            "session_id": stats["sessions"],
            "message": "Session started",
            "latent_dim": codec.latent_dim,
            "gop_size": codec.gop_size,
        }
    )


@app.route("/encode", methods=["POST"])
def encode_frame():
    """
    Encode a single frame.

    Accepts:
        - JSON with "image" (base64) or "image_url"
        - Multipart form with "image" file

    Returns:
        Encoded frame data with statistics
    """
    global stats

    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            if "image" not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            file = request.files["image"]
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif request.is_json:
            data = request.get_json()
            if "image" in data:
                img_data = base64.b64decode(data["image"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return jsonify({"error": "No image data provided"}), 400
        else:
            return jsonify({"error": "Unsupported content type"}), 400

        start = time.perf_counter()
        encoded = codec.encode_frame(frame)
        encode_time = (time.perf_counter() - start) * 1000

        stats["frames_processed"] += 1
        stats["total_bits"] += encoded.bits_used
        stats["total_time_ms"] += encode_time

        return jsonify(
            {
                "frame_num": encoded.frame_num,
                "frame_type": encoded.frame_type,
                "bits_used": encoded.bits_used,
                "encoding_time_ms": round(encode_time, 2),
                "latent_shape": list(encoded.latent.shape),
            }
        )

    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/decode", methods=["POST"])
def decode_frame():
    """
    Decode an encoded frame.

    Accepts:
        JSON with frame_num and optional target_size

    Returns:
        Decoded image as base64
    """
    try:
        data = request.get_json()

        if "frame_num" not in data:
            return jsonify({"error": "frame_num required"}), 400

        frame_num = data["frame_num"]
        target_size = data.get("target_size")

        if frame_num >= len(codec.encoded_frames):
            return jsonify({"error": "Frame not found"}), 404

        encoded = codec.encoded_frames[frame_num]
        target = tuple(target_size) if target_size else None

        start = time.perf_counter()
        decoded = codec.decode_frame(encoded, target)
        decode_time = (time.perf_counter() - start) * 1000

        frame_bgr = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "frame_num": frame_num,
                "decoded_shape": list(decoded.shape),
                "decode_time_ms": round(decode_time, 2),
                "image": img_base64,
            }
        )

    except Exception as e:
        logger.error(f"Decoding error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get encoding statistics."""
    if stats["frames_processed"] == 0:
        return jsonify(
            {
                "frames_processed": 0,
                "avg_bits_per_frame": 0,
                "avg_time_ms": 0,
            }
        )

    return jsonify(
        {
            "frames_processed": stats["frames_processed"],
            "total_bits": stats["total_bits"],
            "avg_bits_per_frame": stats["total_bits"] / stats["frames_processed"],
            "total_time_ms": round(stats["total_time_ms"], 2),
            "avg_time_ms": round(stats["total_time_ms"] / stats["frames_processed"], 2),
            "sessions": stats["sessions"],
            "current_session_frames": len(codec.encoded_frames),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the codec state."""
    codec.reset()
    return jsonify({"message": "Codec reset"})


def create_app(checkpoint_path: str = None):
    """Create and configure the Flask app."""
    load_codec(checkpoint_path)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LeWM-VC Server")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    load_codec(args.checkpoint)

    logger.info(f"Starting LeWM-VC server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
