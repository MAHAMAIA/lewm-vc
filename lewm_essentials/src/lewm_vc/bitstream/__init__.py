"""
Bitstream Package for LeWM-VC

Implements NAL/OBU bitstream serialization and deserialization.
Contains writer and reader classes for video bitstream handling.
"""

from .reader import BitstreamReader
from .writer import BitstreamWriter, NALUnitType

__all__ = [
    "BitstreamWriter",
    "BitstreamReader",
    "NALUnitType",
]
