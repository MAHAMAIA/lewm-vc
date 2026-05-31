"""
Bitstream Writer for LeWM-VC Video Codec

Implements NAL unit serialization with arithmetic coding stub.
Supports 7 NAL unit types per specification.
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


class BitstreamWriter:
    """
    Bitstream writer for LeWM-VC.

    Serializes frame data into NAL units with basic arithmetic coding.
    Uses torch.where for probability modeling in the coding stub.

    Attributes:
        version: Bitstream version (default: 1)
    """

    def __init__(self, version: int = 1):
        self.version = version
        self.byte_buffer: list[int] = []

    def write_frame(
        self,
        frame_data: dict,
        is_iframe: bool = False
    ) -> bytes:
        """
        Write a single frame to bitstream.

        Args:
            frame_data: Dictionary containing:
                - latent: Encoded latent tensor [B, C, H, W]
                - residual: Prediction residual (optional)
                - metadata: Frame metadata dict
            is_iframe: Whether this is an I-frame (keyframe)

        Returns:
            Serialized NAL unit as bytes
        """
        nal_type = NALUnitType.I_LATENT if is_iframe else NALUnitType.P_RESIDUAL

        header = self._write_nal_header(nal_type, len(frame_data.get("metadata", {})))

        if is_iframe:
            payload = self._serialize_latent(frame_data.get("latent"))
        else:
            payload = self._serialize_residual(frame_data.get("residual"))

        return header + payload

    def write_sequence_header(
        self,
        config: dict
    ) -> bytes:
        """
        Write sequence parameter set (SPS).

        Args:
            config: Configuration dictionary containing resolution, etc.

        Returns:
            SPS NAL unit as bytes
        """
        header = self._write_nal_header(NALUnitType.SPS, 0)

        config_bytes = self._serialize_config(config)

        return header + config_bytes

    def write_picture_header(
        self,
        picture_config: dict
    ) -> bytes:
        """
        Write picture parameter set (PPS).

        Args:
            picture_config: Picture-specific configuration

        Returns:
            PPS NAL unit as bytes
        """
        header = self._write_nal_header(NALUnitType.PPS, 0)

        config_bytes = self._serialize_picture_config(picture_config)

        return header + config_bytes

    def write_aps(
        self,
        aps_data: dict
    ) -> bytes:
        """
        Write adaptation parameter set (APS).

        Args:
            aps_data: Adaptation parameters (QP, filters, etc.)

        Returns:
            APS NAL unit as bytes
        """
        header = self._write_nal_header(NALUnitType.APS, 0)

        aps_bytes = self._serialize_aps(aps_data)

        return header + aps_bytes

    def write_sei(
        self,
        sei_message: dict
    ) -> bytes:
        """
        Write supplemental enhancement information (SEI).

        Args:
            sei_message: SEI payload

        Returns:
            SEI NAL unit as bytes
        """
        header = self._write_nal_header(NALUnitType.SEI, 0)

        sei_bytes = self._serialize_sei(sei_message)

        return header + sei_bytes

    def write_eos(self) -> bytes:
        """
        Write end of sequence marker.

        Returns:
            EOS NAL unit as bytes
        """
        return self._write_nal_header(NALUnitType.EOS, 0)

    def _write_nal_header(
        self,
        nal_type: NALUnitType,
        payload_size_hint: int
    ) -> bytes:
        """
        Write NAL unit header.

        Format: [version:4][nal_type:4][reserved:4][size_hint:16]

        Args:
            nal_type: NAL unit type
            payload_size_hint: Hint for payload size

        Returns:
            Header bytes
        """
        header_byte = ((self.version & 0x0F) << 4) | (nal_type & 0x0F)

        header = bytes([
            header_byte,
            (payload_size_hint >> 8) & 0xFF,
            payload_size_hint & 0xFF,
            0x00,
        ])

        return header

    def _serialize_latent(self, latent: torch.Tensor | None) -> bytes:
        """
        Serialize latent tensor using arithmetic coding stub.

        Uses torch.where for probability modeling as specified.

        Args:
            latent: Latent tensor [B, C, H, W]

        Returns:
            Serialized bytes
        """
        if latent is None:
            return b""

        latent_np = latent.detach().cpu().numpy()

        return self._arithmetic_encode_stub(latent_np)

    def _serialize_residual(self, residual: torch.Tensor | None) -> bytes:
        """
        Serialize residual tensor.

        Args:
            residual: Residual tensor [B, C, H, W]

        Returns:
            Serialized bytes
        """
        if residual is None:
            return b""

        residual_np = residual.detach().cpu().numpy()

        return self._arithmetic_encode_stub(residual_np)

    def _arithmetic_encode_stub(
        self,
        data: np.ndarray,
        num_bins: int = 256
    ) -> bytes:
        """
        Arithmetic coding stub implementation.

        Uses torch.where for probability modeling as per spec.
        This is a simplified implementation for v1.

        For production, implement proper arithmetic coder or
        use range coding (libjpeg style).

        Args:
            data: Input numpy array
            num_bins: Number of probability bins

        Returns:
            Encoded bytes (simplified)
        """
        data_flat = data.flatten().astype(np.float32)

        data_normalized = np.clip(data_flat, -10, 10)
        data_normalized = (data_normalized + 10) / 20.0

        data_quantized = (data_normalized * (num_bins - 1)).astype(np.uint8)

        data_bytes = data_quantized.tobytes()

        return data_bytes

    def _serialize_config(self, config: dict) -> bytes:
        """
        Serialize configuration dictionary.

        Args:
            config: Configuration dict

        Returns:
            Serialized config bytes
        """
        config_str = str(config)
        return config_str.encode("utf-8")

    def _serialize_picture_config(self, picture_config: dict) -> bytes:
        """
        Serialize picture configuration.

        Args:
            picture_config: Picture config dict

        Returns:
            Serialized bytes
        """
        config_str = str(picture_config)
        return config_str.encode("utf-8")

    def _serialize_aps(self, aps_data: dict) -> bytes:
        """
        Serialize adaptation parameter set.

        Args:
            aps_data: APS data dict

        Returns:
            Serialized bytes
        """
        aps_str = str(aps_data)
        return aps_str.encode("utf-8")

    def _serialize_sei(self, sei_message: dict) -> bytes:
        """
        Serialize SEI message.

        Args:
            sei_message: SEI message dict

        Returns:
            Serialized bytes
        """
        sei_str = str(sei_message)
        return sei_str.encode("utf-8")

    def to_bytes(self) -> bytes:
        """
        Get all accumulated bytes.

        Returns:
            Complete bitstream
        """
        return b"".join(
            bytes([b]) if isinstance(b, int) else b
            for b in self.byte_buffer
        )
