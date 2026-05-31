"""
Bitstream Reader for LeWM-VC Video Codec

Implements NAL unit parsing and deserialization.
Complements BitstreamWriter for round-trip encoding/decoding.
"""

from enum import IntEnum

import numpy as np
import torch


class NALUnitType(IntEnum):
    """NAL unit types for LeWM-VC bitstream."""
    SPS = 0
    PPS = 1
    APS = 2
    I_LATENT = 3
    P_RESIDUAL = 4
    SEI = 5
    EOS = 6


class BitstreamReader:
    """
    Bitstream reader for LeWM-VC.

    Parses NAL units from bitstream and reconstructs frame data.
    Complements BitstreamWriter for round-trip operations.

    Attributes:
        version: Bitstream version (default: 1)
    """

    def __init__(self, version: int = 1):
        self.version = version
        self.position: int = 0

    def read_frame(
        self,
        stream: bytes
    ) -> dict:
        """
        Read a single frame from bitstream.

        Args:
            stream: Input byte stream

        Returns:
            Dictionary containing:
                - nal_type: NAL unit type
                - latent: Decoded latent tensor (if I_LATENT)
                - residual: Decoded residual tensor (if P_RESIDUAL)
                - metadata: Frame metadata
        """
        nal_type, payload = self._parse_nal(stream)

        if nal_type == NALUnitType.I_LATENT:
            latent = self._deserialize_latent(payload)
            return {
                "nal_type": nal_type,
                "latent": latent,
                "metadata": {},
            }
        elif nal_type == NALUnitType.P_RESIDUAL:
            residual = self._deserialize_residual(payload)
            return {
                "nal_type": nal_type,
                "residual": residual,
                "metadata": {},
            }
        else:
            return {
                "nal_type": nal_type,
                "payload": payload,
                "metadata": {},
            }

    def read_sequence_header(self, stream: bytes) -> dict:
        """
        Read sequence parameter set (SPS).

        Args:
            stream: Input byte stream

        Returns:
            Configuration dictionary
        """
        nal_type, payload = self._parse_nal(stream)

        if nal_type != NALUnitType.SPS:
            raise ValueError(f"Expected SPS NAL unit, got {nal_type}")

        config_str = payload.decode("utf-8")
        config = eval(config_str)

        return {"nal_type": nal_type, "config": config}

    def read_picture_header(self, stream: bytes) -> dict:
        """
        Read picture parameter set (PPS).

        Args:
            stream: Input byte stream

        Returns:
            Picture configuration dictionary
        """
        nal_type, payload = self._parse_nal(stream)

        if nal_type != NALUnitType.PPS:
            raise ValueError(f"Expected PPS NAL unit, got {nal_type}")

        config_str = payload.decode("utf-8")
        config = eval(config_str)

        return {"nal_type": nal_type, "config": config}

    def read_aps(self, stream: bytes) -> dict:
        """
        Read adaptation parameter set (APS).

        Args:
            stream: Input byte stream

        Returns:
            APS data dictionary
        """
        nal_type, payload = self._parse_nal(stream)

        if nal_type != NALUnitType.APS:
            raise ValueError(f"Expected APS NAL unit, got {nal_type}")

        aps_str = payload.decode("utf-8")
        aps_data = eval(aps_str)

        return {"nal_type": nal_type, "aps_data": aps_data}

    def read_sei(self, stream: bytes) -> dict:
        """
        Read supplemental enhancement information (SEI).

        Args:
            stream: Input byte stream

        Returns:
            SEI message dictionary
        """
        nal_type, payload = self._parse_nal(stream)

        if nal_type != NALUnitType.SEI:
            raise ValueError(f"Expected SEI NAL unit, got {nal_type}")

        sei_str = payload.decode("utf-8")
        sei_message = eval(sei_str)

        return {"nal_type": nal_type, "sei_message": sei_message}

    def read_eos(self, stream: bytes) -> bool:
        """
        Read end of sequence marker.

        Args:
            stream: Input byte stream

        Returns:
            True if EOS marker found
        """
        nal_type, _ = self._parse_nal(stream)
        return nal_type == NALUnitType.EOS

    def _parse_nal(self, stream: bytes) -> tuple[NALUnitType, bytes]:
        """
        Parse NAL unit header and extract payload.

        Args:
            stream: Input byte stream

        Returns:
            Tuple of (NAL type, payload bytes)
        """
        if len(stream) < 4:
            raise ValueError("Stream too short for NAL header")

        header = stream[:4]
        payload = stream[4:]

        header_byte = header[0]
        nal_type = header_byte & 0x0F

        version = (header_byte >> 4) & 0x0F
        if version != self.version:
            raise ValueError(f"Version mismatch: expected {self.version}, got {version}")

        try:
            nal_unit_type = NALUnitType(nal_type)
        except ValueError as err:
            raise ValueError(f"Unknown NAL unit type: {nal_type}") from err

        return nal_unit_type, payload

    def _deserialize_latent(self, payload: bytes) -> torch.Tensor:
        """
        Deserialize latent tensor.

        Args:
            payload: Serialized payload bytes

        Returns:
            Decoded latent tensor
        """
        return self._arithmetic_decode_stub(payload)

    def _deserialize_residual(self, payload: bytes) -> torch.Tensor:
        """
        Deserialize residual tensor.

        Args:
            payload: Serialized payload bytes

        Returns:
            Decoded residual tensor
        """
        return self._arithmetic_decode_stub(payload)

    def _arithmetic_decode_stub(
        self,
        data_bytes: bytes,
        shape: tuple = (1, 192, 16, 16),
        num_bins: int = 256
    ) -> torch.Tensor:
        """
        Arithmetic decoding stub implementation.

        Complements the writer stub - inverse operation.

        Args:
            data_bytes: Encoded bytes
            shape: Expected output shape
            num_bins: Number of probability bins

        Returns:
            Decoded tensor
        """
        data_np = np.frombuffer(data_bytes, dtype=np.uint8)

        data_normalized = data_np.astype(np.float32) / (num_bins - 1)

        data_denormalized = data_normalized * 20.0 - 10.0

        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if len(data_denormalized) < total_elements:
            data_denormalized = np.pad(
                data_denormalized,
                (0, total_elements - len(data_denormalized)),
                mode="constant",
                constant_values=0
            )
        else:
            data_denormalized = data_denormalized[:total_elements]

        data_tensor = torch.from_numpy(
            data_denormalized.reshape(shape)
        ).float()

        return data_tensor

    def set_position(self, position: int) -> None:
        """
        Set current position in stream.

        Args:
            position: Byte position
        """
        self.position = position

    def get_position(self) -> int:
        """
        Get current position in stream.

        Returns:
            Current byte position
        """
        return self.position
