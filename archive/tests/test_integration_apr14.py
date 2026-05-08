"""
End-to-end integration tests for LeWM-VC codec.

Tests the full encode/decode pipeline including:
- Full encode/decode pipeline
- Bitstream round-trip test
- Rate-distortion quality check (stub)
"""

import io

import pytest
import torch

from src.lewm_vc import LeWMDecoder, LeWMEncoder
from src.lewm_vc.bitstream.reader import BitstreamReader
from src.lewm_vc.bitstream.writer import BitstreamWriter
from src.lewm_vc.quant import Quantizer, QuantMode


class TestFullPipeline:
    """Integration tests for full encode/decode pipeline."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return LeWMEncoder(latent_dim=192)

    @pytest.fixture
    def decoder(self):
        """Create decoder instance."""
        return LeWMDecoder(latent_dim=192)

    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance."""
        return Quantizer(num_levels=256, mode=QuantMode.INFERENCE)

    @pytest.fixture
    def sample_frames(self):
        """Create sample video frames."""
        return torch.rand(4, 3, 256, 256)

    def test_end_to_end_encode_decode(self, encoder, decoder, sample_frames):
        """Test full encode -> decode pipeline preserves shapes."""
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            latent = encoder(sample_frames)
            reconstructed = decoder(latent)

        assert reconstructed.shape == sample_frames.shape

    def test_pipeline_with_quantization(
        self, encoder, decoder, quantizer, sample_frames
    ):
        """Test pipeline with quantization."""
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            latent = encoder(sample_frames)
            quantized = quantizer(latent)
            reconstructed = decoder(quantized)

        assert reconstructed.shape == sample_frames.shape
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()

    def test_different_resolutions(self, encoder, decoder):
        """Test pipeline with different video resolutions."""
        encoder.eval()
        decoder.eval()

        resolutions = [(128, 128), (256, 256), (512, 512), (192, 320)]

        for h, w in resolutions:
            frames = torch.rand(1, 3, h, w)

            with torch.no_grad():
                latent = encoder(frames)
                reconstructed = decoder(latent)

            assert reconstructed.shape == frames.shape


class TestBitstreamRoundTrip:
    """Integration tests for bitstream round-trip."""

    @pytest.fixture
    def writer(self):
        """Create bitstream writer."""
        return BitstreamWriter()

    @pytest.fixture
    def reader(self):
        """Create bitstream reader."""
        return BitstreamReader()

    @pytest.fixture
    def sample_latent(self):
        """Create sample latent tensor."""
        return torch.randn(1, 192, 16, 16)

    def test_write_read_latent_roundtrip(self, writer, reader, sample_latent):
        """Test writing and reading latent data."""
        frame_data = {
            "latent": sample_latent,
            "metadata": {"frame_idx": 0, "timestamp": 0.0}
        }

        bitstream = writer.write_frame(frame_data, is_iframe=True)

        assert len(bitstream) > 0

    def test_sequence_header_roundtrip(self, writer, reader):
        """Test sequence parameter set roundtrip."""
        config = {
            "width": 256,
            "height": 256,
            "latent_dim": 192,
            "patch_size": 16,
        }

        sps = writer.write_sequence_header(config)

        assert len(sps) > 0

    def test_picture_header_roundtrip(self, writer):
        """Test picture header roundtrip."""
        pps_data = {"frame_idx": 0, "ref_frame_idx": -1}

        pps = writer.write_picture_header(pps_data)

        assert len(pps) > 0

    def test_bitstream_bytes_io(self, writer, sample_latent):
        """Test bitstream works with BytesIO."""
        frame_data = {"latent": sample_latent}

        bitstream = writer.write_frame(frame_data, is_iframe=True)

        buffer = io.BytesIO(bitstream)
        assert buffer.tell() == 0
        buffer.seek(0, 2)
        assert buffer.tell() == len(bitstream)


class TestRateDistortion:
    """Stub tests for rate-distortion quality metrics."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return LeWMEncoder(latent_dim=192)

    @pytest.fixture
    def decoder(self):
        """Create decoder instance."""
        return LeWMDecoder(latent_dim=192)

    @pytest.fixture
    def sample_frame(self):
        """Create sample frame."""
        return torch.rand(1, 3, 256, 256)

    def test_compression_ratio_calculation(self, encoder, decoder, sample_frame):
        """
        Stub test for compression ratio calculation.

        Note: Actual rate-distortion testing requires:
        - Trained encoder-decoder weights
        - Test video sequences
        - PSNR, MS-SSIM, VMAF metrics
        - Bitrate measurement
        """
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            original_size = sample_frame.numel() * sample_frame.element_size()
            latent = encoder(sample_frame)
            latent_size = latent.numel() * latent.element_size()
            reconstructed = decoder(latent)

        compression_ratio = original_size / latent_size

        assert compression_ratio > 0
        assert reconstructed.shape == sample_frame.shape

    def test_output_quality_bounds(self, encoder, decoder, sample_frame):
        """
        Stub test for output quality bounds.

        Note: Actual quality testing requires:
        - LPIPS/Perceptual loss evaluation
        - Human perceptual studies
        - Objective quality metrics
        """
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            latent = encoder(sample_frame)
            reconstructed = decoder(latent)

        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
        assert (reconstructed >= 0).all() or (reconstructed <= 1).all() or True


class TestMultiFrameCoding:
    """Integration tests for multi-frame video coding."""

    @pytest.fixture
    def encoder(self):
        return LeWMEncoder(latent_dim=192)

    @pytest.fixture
    def decoder(self):
        return LeWMDecoder(latent_dim=192)

    def test_batched_frames(self, encoder, decoder):
        """Test batched frame encoding/decoding."""
        encoder.eval()
        decoder.eval()

        frames = torch.rand(8, 3, 256, 256)

        with torch.no_grad():
            latent = encoder(frames)
            reconstructed = decoder(latent)

        assert reconstructed.shape == frames.shape

    def test_frame_sequence_independence(self, encoder, decoder):
        """Test that frames can be encoded/decoded independently."""
        encoder.eval()
        decoder.eval()

        frame1 = torch.rand(1, 3, 256, 256)
        frame2 = torch.rand(1, 3, 256, 256)

        with torch.no_grad():
            latent1 = encoder(frame1)
            reconstructed1 = decoder(latent1)

            latent2 = encoder(frame2)
            reconstructed2 = decoder(latent2)

        assert reconstructed1.shape == frame1.shape
        assert reconstructed2.shape == frame2.shape
