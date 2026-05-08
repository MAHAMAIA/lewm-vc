"""
Unit tests for Bitstream Module

Tests bitstream writer and reader with round-trip encoding/decoding.
"""

import pytest
import torch

from src.lewm_vc.bitstream import BitstreamReader, BitstreamWriter, NALUnitType


class TestBitstreamWriter:
    """Test suite for BitstreamWriter class."""

    @pytest.fixture
    def writer(self):
        """Create writer instance."""
        return BitstreamWriter(version=1)

    def test_initialization(self, writer):
        """Test writer initializes correctly."""
        assert writer.version == 1

    def test_write_frame_iframe(self, writer):
        """Test writing I-frame."""
        latent = torch.randn(1, 192, 16, 16)
        frame_data = {"latent": latent, "metadata": {}}

        nal_bytes = writer.write_frame(frame_data, is_iframe=True)

        assert len(nal_bytes) > 4

    def test_write_frame_pframe(self, writer):
        """Test writing P-frame."""
        residual = torch.randn(1, 192, 16, 16)
        frame_data = {"residual": residual, "metadata": {}}

        nal_bytes = writer.write_frame(frame_data, is_iframe=False)

        assert len(nal_bytes) > 4

    def test_write_sequence_header(self, writer):
        """Test writing SPS."""
        config = {"width": 1920, "height": 1080, "fps": 30}

        nal_bytes = writer.write_sequence_header(config)

        assert len(nal_bytes) > 4
        assert nal_bytes[0] & 0x0F == NALUnitType.SPS

    def test_write_picture_header(self, writer):
        """Test writing PPS."""
        picture_config = {"qp": 28, "ref_frames": 4}

        nal_bytes = writer.write_picture_header(picture_config)

        assert len(nal_bytes) > 4
        assert nal_bytes[0] & 0x0F == NALUnitType.PPS

    def test_write_aps(self, writer):
        """Test writing APS."""
        aps_data = {"qp_offset": 0, "lf": True}

        nal_bytes = writer.write_aps(aps_data)

        assert len(nal_bytes) > 4
        assert nal_bytes[0] & 0x0F == NALUnitType.APS

    def test_write_sei(self, writer):
        """Test writing SEI."""
        sei_message = {"payload_type": 5, "data": "test"}

        nal_bytes = writer.write_sei(sei_message)

        assert len(nal_bytes) > 4
        assert nal_bytes[0] & 0x0F == NALUnitType.SEI

    def test_write_eos(self, writer):
        """Test writing EOS."""
        nal_bytes = writer.write_eos()

        assert len(nal_bytes) == 4
        assert nal_bytes[0] & 0x0F == NALUnitType.EOS


class TestBitstreamReader:
    """Test suite for BitstreamReader class."""

    @pytest.fixture
    def reader(self):
        """Create reader instance."""
        return BitstreamReader(version=1)

    def test_initialization(self, reader):
        """Test reader initializes correctly."""
        assert reader.version == 1

    def test_read_eos(self, reader):
        """Test reading EOS marker."""
        writer = BitstreamWriter(version=1)
        eos_bytes = writer.write_eos()

        is_eos = reader.read_eos(eos_bytes)

        assert is_eos is True

    def test_roundtrip_iframe(self, reader):
        """Test round-trip I-frame encoding/decoding."""
        writer = BitstreamWriter(version=1)
        latent = torch.randn(1, 192, 16, 16)
        frame_data = {"latent": latent, "metadata": {}}

        nal_bytes = writer.write_frame(frame_data, is_iframe=True)
        decoded = reader.read_frame(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.I_LATENT
        assert "latent" in decoded

    def test_roundtrip_pframe(self, reader):
        """Test round-trip P-frame encoding/decoding."""
        writer = BitstreamWriter(version=1)
        residual = torch.randn(1, 192, 16, 16)
        frame_data = {"residual": residual, "metadata": {}}

        nal_bytes = writer.write_frame(frame_data, is_iframe=False)
        decoded = reader.read_frame(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.P_RESIDUAL
        assert "residual" in decoded

    def test_roundtrip_sps(self, reader):
        """Test round-trip SPS encoding/decoding."""
        writer = BitstreamWriter(version=1)
        config = {"width": 1920, "height": 1080}

        nal_bytes = writer.write_sequence_header(config)
        decoded = reader.read_sequence_header(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.SPS

    def test_roundtrip_pps(self, reader):
        """Test round-trip PPS encoding/decoding."""
        writer = BitstreamWriter(version=1)
        picture_config = {"qp": 28}

        nal_bytes = writer.write_picture_header(picture_config)
        decoded = reader.read_picture_header(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.PPS

    def test_roundtrip_aps(self, reader):
        """Test round-trip APS encoding/decoding."""
        writer = BitstreamWriter(version=1)
        aps_data = {"qp_offset": 0}

        nal_bytes = writer.write_aps(aps_data)
        decoded = reader.read_aps(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.APS

    def test_roundtrip_sei(self, reader):
        """Test round-trip SEI encoding/decoding."""
        writer = BitstreamWriter(version=1)
        sei_message = {"payload_type": 5}

        nal_bytes = writer.write_sei(sei_message)
        decoded = reader.read_sei(nal_bytes)

        assert decoded["nal_type"] == NALUnitType.SEI


class TestRoundTrip:
    """End-to-end round-trip tests."""

    def test_frame_sequence_roundtrip(self):
        """Test sequence of frames round-trips correctly."""
        writer = BitstreamWriter(version=1)
        reader = BitstreamReader(version=1)

        config = {"width": 1920, "height": 1080, "fps": 30}
        sps_bytes = writer.write_sequence_header(config)

        picture_config = {"qp": 28}
        pps_bytes = writer.write_picture_header(picture_config)

        frames = []
        for i in range(10):
            is_iframe = i % 10 == 0
            if is_iframe:
                latent = torch.randn(1, 192, 16, 16)
                frame_data = {"latent": latent, "metadata": {"frame_id": i}}
            else:
                residual = torch.randn(1, 192, 16, 16)
                frame_data = {"residual": residual, "metadata": {"frame_id": i}}

            frame_bytes = writer.write_frame(frame_data, is_iframe=is_iframe)
            frames.append(frame_bytes)

        decoded_sps = reader.read_sequence_header(sps_bytes)
        assert decoded_sps["nal_type"] == NALUnitType.SPS

        decoded_pps = reader.read_picture_header(pps_bytes)
        assert decoded_pps["nal_type"] == NALUnitType.PPS

        for _i, frame_bytes in enumerate(frames):
            decoded_frame = reader.read_frame(frame_bytes)
            assert "nal_type" in decoded_frame


class TestNALUnitType:
    """Test suite for NAL unit types."""

    def test_nal_types(self):
        """Test all NAL unit types are defined."""
        assert NALUnitType.SPS == 0
        assert NALUnitType.PPS == 1
        assert NALUnitType.APS == 2
        assert NALUnitType.I_LATENT == 3
        assert NALUnitType.P_RESIDUAL == 4
        assert NALUnitType.SEI == 5
        assert NALUnitType.EOS == 6


class TestVersionHandling:
    """Test version handling."""

    def test_version_mismatch_raises(self):
        """Test version mismatch raises error."""
        writer = BitstreamWriter(version=1)
        reader = BitstreamReader(version=2)

        latent = torch.randn(1, 192, 16, 16)
        frame_data = {"latent": latent}
        nal_bytes = writer.write_frame(frame_data, is_iframe=True)

        with pytest.raises(ValueError, match="Version mismatch"):
            reader.read_frame(nal_bytes)
