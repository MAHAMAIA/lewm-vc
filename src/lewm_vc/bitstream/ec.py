"""
Bitstream entropy coding for LeWM-VC.

Uses zlib compression on quantized latent indices. This is not as efficient
as rANS arithmetic coding but is correct and simple. rANS can replace zlib
once a tested library (e.g. torchac, constriction) is integrated.

Compression ratio: zlib typically achieves 60-80% of theoretical entropy
for typical latent distributions, vs 98-99% for rANS. Acceptable for pilots.
"""

import struct
import zlib
from enum import IntEnum

import numpy as np

# ---------------------------------------------------------------------------
# NAL unit types and structure
# ---------------------------------------------------------------------------

SYNC_BYTE = 0x4C  # 'L'


class NALType(IntEnum):
    SPS = 0
    PPS = 1
    I_FRAME = 3
    P_FRAME = 4
    EOS = 6


def pack_nal(nal_type: int, payload: bytes) -> bytes:
    """Pack a NAL unit: sync(1) + type(1) + size(2 big-endian) + payload."""
    size = len(payload)
    header = struct.pack(">BBH", SYNC_BYTE, nal_type, size)
    return header + payload


def unpack_nal(stream: bytes, offset: int = 0) -> tuple[int, bytes, int]:
    """Unpack a NAL unit from stream at offset. Returns (type, payload, next_offset)."""
    if offset + 4 > len(stream):
        raise ValueError("Stream too short for NAL header")
    sync, nal_type, size = struct.unpack_from(">BBH", stream, offset)
    if sync != SYNC_BYTE:
        raise ValueError(f"Bad sync byte: {sync:#x}")
    payload_start = offset + 4
    payload_end = payload_start + size
    if payload_end > len(stream):
        raise ValueError(f"NAL payload exceeds stream: {payload_end} > {len(stream)}")
    return nal_type, stream[payload_start:payload_end], payload_end


# ---------------------------------------------------------------------------
# Compression with zlib (placeholder for rANS)
# ---------------------------------------------------------------------------


def compress_indices(indices: np.ndarray, cdf: np.ndarray | None = None) -> bytes:
    """Compress quantized latent indices using zlib.

    Args:
        indices: uint8 array of quantized latent values
        cdf: ignored (zlib is CDF-agnostic). Used for rANS replacement.

    Returns:
        Compressed bytes
    """
    raw = indices.tobytes()
    return zlib.compress(raw, level=6)


def decompress_indices(
    compressed: bytes, size: int, dtype=np.uint8, cdf: np.ndarray | None = None
) -> np.ndarray:
    """Decompress to original indices array."""
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(-1)[:size]


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def quantize_latent(latent: np.ndarray, step: float = 2.0 / 255) -> np.ndarray:
    """Quantize latent tensor to uint8 indices.

    Args:
        latent: float32 array
        step: quantization step size (default matches training setup)

    Returns:
        uint8 array of same shape
    """
    return (np.round(latent / step).clip(-128, 127).astype(np.int16) + 128).astype(np.uint8)


def dequantize_indices(
    indices: np.ndarray, step: float = 2.0 / 255, shape: tuple | None = None
) -> np.ndarray:
    """Dequantize uint8 indices back to float.

    Args:
        indices: uint8 array
        step: quantization step size
        shape: reshape if provided

    Returns:
        float32 array
    """
    vals = indices.astype(np.float32) - 128
    if shape is not None:
        vals = vals.reshape(shape)
    return vals * step


# ---------------------------------------------------------------------------
# Bitstream frame encoder (produces actual .lewm file)
# ---------------------------------------------------------------------------


def encode_frame(is_iframe: bool, latent_np: np.ndarray) -> bytes:
    """Encode one frame's latent/residual into a NAL unit.

    Args:
        is_iframe: True for I-frame (quantized latent), False for P-frame (residual)
        nal_type: NAL type for the frame

    Returns:
        NAL unit bytes
    """
    indices = quantize_latent(latent_np)
    compressed = compress_indices(indices)
    nal_type = NALType.I_FRAME if is_iframe else NALType.P_FRAME
    return pack_nal(nal_type, compressed)


def decode_frame(nal: bytes, shape: tuple) -> np.ndarray:
    """Decode one frame's latent/residual from a NAL unit.

    Args:
        nal: NAL unit bytes
        shape: expected shape of the reconstructed latent (e.g., (192, 16, 16))

    Returns:
        Dequantized float32 array
    """
    _, payload, _ = unpack_nal(nal, 0)
    indices = decompress_indices(payload, size=np.prod(shape))
    return dequantize_indices(indices, shape=shape)


# ---------------------------------------------------------------------------
# Full bitstream: sequence header + frames
# ---------------------------------------------------------------------------


def write_sequence_header(width: int, height: int, latent_dim: int = 192) -> bytes:
    """Write SPS NAL unit with video parameters."""
    params = struct.pack(">HHH", width, height, latent_dim)
    return pack_nal(NALType.SPS, params)


def read_sequence_header(stream: bytes) -> tuple[int, int, int]:
    """Read SPS and return (width, height, latent_dim)."""
    _, payload, _ = unpack_nal(stream, 0)
    width, height, latent_dim = struct.unpack_from(">HHH", payload, 0)
    return width, height, latent_dim


def write_eos() -> bytes:
    """Write end-of-stream marker."""
    return pack_nal(NALType.EOS, b"")
