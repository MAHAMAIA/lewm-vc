"""
Bitstream Package for LeWM-VC

Implements NAL unit-based bitstream with zlib entropy coding.
"""

from .ec import (
    NALType,
    compress_indices,
    decompress_indices,
    quantize_latent,
    dequantize_indices,
    encode_frame,
    decode_frame,
    write_sequence_header,
    read_sequence_header,
    write_eos,
    pack_nal,
    unpack_nal,
)

__all__ = [
    "NALType",
    "compress_indices",
    "decompress_indices",
    "quantize_latent",
    "dequantize_indices",
    "encode_frame",
    "decode_frame",
    "write_sequence_header",
    "read_sequence_header",
    "write_eos",
    "pack_nal",
    "unpack_nal",
]
