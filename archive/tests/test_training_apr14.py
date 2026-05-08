"""
Unit tests for Training Pipeline.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scripts.train import LeWMTrainer, TrainingPhase, VideoDataset


def test_training_phase_enum():
    """Test TrainingPhase enum values."""
    assert TrainingPhase.DECODER_WARMUP == 0
    assert TrainingPhase.JOINT_RD == 1
    assert TrainingPhase.QAT == 2
    assert TrainingPhase.DISTILLATION == 3


def test_training_phase_names():
    """Test TrainingPhase name mapping."""
    assert TrainingPhase.get_name(0) == "decoder_warmup"
    assert TrainingPhase.get_name(1) == "joint_rd"
    assert TrainingPhase.get_name(2) == "qat"
    assert TrainingPhase.get_name(3) == "distillation"


def test_training_phase_durations():
    """Test TrainingPhase duration values."""
    assert TrainingPhase.get_duration_hours(0) == 24
    assert TrainingPhase.get_duration_hours(1) == 72
    assert TrainingPhase.get_duration_hours(2) == 48
    assert TrainingPhase.get_duration_hours(3) == 24


def test_video_dataset_initialization():
    """Test VideoDataset initializes correctly."""
    config = {
        "train": {
            "videos": [
                {"path": "/path/to/video1.mp4", "fps": 30, "frame_count": 300},
                {"path": "/path/to/video2.mp4", "fps": 24, "frame_count": 240},
            ]
        },
        "val": {"videos": []},
        "test": {"videos": []},
    }

    dataset = VideoDataset(config, split="train", sequence_length=16)
    assert len(dataset) == 2


def test_video_dataset_getitem():
    """Test VideoDataset __getitem__."""
    config = {
        "train": {"videos": [{"path": "/path/to/video.mp4", "fps": 30}]},
        "val": {"videos": []},
        "test": {"videos": []},
    }

    dataset = VideoDataset(config, split="train", sequence_length=8)

    item = dataset[0]
    assert "frames" in item
    assert "path" in item
    assert item["frames"].shape == (8, 3, 256, 256)


def test_lewm_trainer_initialization():
    """Test LeWMTrainer initializes correctly."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {
        "logging": {"tensorboard": False},
        "checkpoint": {"dir": "checkpoints"},
    }

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    assert trainer.current_phase == 0
    assert trainer.writer is None


def test_lewm_trainer_phase_switch():
    """Test phase switching logic."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {"logging": {"tensorboard": False}, "checkpoint": {"dir": "/tmp"}}

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    assert trainer.current_phase == 0

    trainer.switch_phase(1)
    assert trainer.current_phase == 1
    assert trainer.phase_step == 0


def test_lewm_trainer_invalid_phase():
    """Test invalid phase raises error."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {"logging": {"tensorboard": False}, "checkpoint": {"dir": "/tmp"}}

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    with pytest.raises(ValueError):
        trainer.switch_phase(5)


def test_compute_loss_output_shape():
    """Test loss computation produces correct output structure."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {"logging": {"tensorboard": False}, "checkpoint": {"dir": "/tmp"}}

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    frames = torch.randn(1, 4, 3, 256, 256)

    losses = trainer.compute_loss(frames, lambda_val=0.1)

    assert "total_loss" in losses
    assert "rate_loss" in losses
    assert "distortion_loss" in losses
    assert "mse_loss" in losses
    assert "lpips_loss" in losses
    assert "surprise_loss" in losses
    assert "rate_bits" in losses


def test_trainer_save_checkpoint(tmp_path):
    """Test checkpoint saving."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {
        "logging": {"tensorboard": False},
        "checkpoint": {"dir": str(tmp_path)},
    }

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    checkpoint_path = trainer.save_checkpoint("test")

    assert os.path.exists(checkpoint_path)


def test_trainer_load_checkpoint(tmp_path):
    """Test checkpoint loading."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {
        "logging": {"tensorboard": False},
        "checkpoint": {"dir": str(tmp_path)},
    }

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    checkpoint_path = trainer.save_checkpoint("test")
    trainer.global_step = 100
    trainer.phase_step = 50
    checkpoint_path = trainer.save_checkpoint("test")

    encoder2 = LeWMEncoder()
    predictor2 = LeWMPredictor()
    decoder2 = LeWMDecoder()
    entropy_model2 = HyperpriorEntropy()
    quantizer2 = Quantizer()
    rate_controller2 = RateController()

    trainer2 = LeWMTrainer(
        encoder=encoder2,
        predictor=predictor2,
        decoder=decoder2,
        entropy_model=entropy_model2,
        quantizer=quantizer2,
        rate_controller=rate_controller2,
        config=config,
        device="cpu",
    )

    trainer2.load_checkpoint(checkpoint_path)

    assert trainer2.global_step == 100
    assert trainer2.phase_step == 50


def test_trainer_close():
    """Test trainer close cleans up resources."""
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.rate_control import RateController

    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {
        "logging": {"tensorboard": False},
        "checkpoint": {"dir": "/tmp"},
    }

    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )

    trainer.close()
