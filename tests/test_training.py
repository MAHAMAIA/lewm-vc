"""Tests for Training Pipeline."""

import torch

from src.lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
from src.lewm_vc.entropy import HyperpriorEntropy
from src.lewm_vc.quant import Quantizer
from src.lewm_vc.utils.rate_control import RateController
from src.scripts.train import LeWMTrainer, TrainingPhase


def _make_trainer(tmp_path=None, config_overrides=None):
    encoder = LeWMEncoder()
    predictor = LeWMPredictor()
    decoder = LeWMDecoder()
    entropy_model = HyperpriorEntropy()
    quantizer = Quantizer()
    rate_controller = RateController()

    config = {"logging": {"tensorboard": False}, "checkpoint": {"dir": str(tmp_path or "/tmp")}}
    if config_overrides:
        config.update(config_overrides)

    return LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device="cpu",
    )


def test_training_phase_enum():
    assert TrainingPhase.DECODER_WARMUP == 0
    assert TrainingPhase.JOINT_RD == 1
    assert TrainingPhase.QAT == 2
    assert TrainingPhase.DISTILLATION == 3


def test_training_phase_names():
    assert TrainingPhase.get_name(0) == "warmup"
    assert TrainingPhase.get_name(1) == "joint_rd"
    assert TrainingPhase.get_name(2) == "qat"
    assert TrainingPhase.get_name(3) == "distillation"
    assert TrainingPhase.get_name(4) == "cooldown"


def test_lewm_trainer_initialization():
    trainer = _make_trainer()
    assert trainer.current_phase == 0
    assert trainer.writer is None


def test_lewm_trainer_phase_switch():
    trainer = _make_trainer()
    assert trainer.current_phase == 0
    trainer.switch_phase(1)
    assert trainer.current_phase == 1
    assert trainer.phase_step == 0


def test_lewm_trainer_switch_back():
    trainer = _make_trainer()
    trainer.switch_phase(2)
    assert trainer.current_phase == 2
    trainer.switch_phase(0)
    assert trainer.current_phase == 0


def test_compute_loss_output_shape():
    trainer = _make_trainer()
    frames = torch.randn(1, 4, 3, 256, 256)
    losses = trainer.compute_loss(frames)
    expected_keys = {
        "total_loss",
        "rate_loss",
        "distortion_loss",
        "mse_loss",
        "lpips_loss",
        "jepa_loss",
        "sigreg_loss",
        "surprise_loss",
    }
    assert expected_keys.issubset(losses.keys())
    for v in losses.values():
        assert isinstance(v, torch.Tensor)
        assert v.ndim == 0


def test_trainer_save_checkpoint(tmp_path):
    trainer = _make_trainer(tmp_path)
    optimizer = torch.optim.AdamW(
        [p for m in trainer.models.values() for p in m.parameters() if p.requires_grad],
        lr=1e-4,
    )
    path = trainer.save_checkpoint("test", optimizer)
    assert path.endswith(".pt")
    assert tmp_path.joinpath("lambda_0.05", "test.pt").exists()


def test_trainer_load_checkpoint(tmp_path):
    trainer = _make_trainer(tmp_path)
    optimizer = torch.optim.AdamW(
        [p for m in trainer.models.values() for p in m.parameters() if p.requires_grad],
        lr=1e-4,
    )
    trainer.save_checkpoint("test", optimizer)
    trainer.global_step = 100
    trainer.phase_step = 50
    trainer.save_checkpoint("test2", optimizer)

    trainer2 = _make_trainer(tmp_path)
    path = tmp_path / "lambda_0.05" / "test2.pt"
    trainer2.load_checkpoint(str(path))
    assert trainer2.global_step == 100
    assert trainer2.phase_step == 50


def test_trainer_close():
    trainer = _make_trainer()
    trainer.close()


def test_train_step():
    trainer = _make_trainer()
    optimizer = torch.optim.AdamW(
        [p for m in trainer.models.values() for p in m.parameters() if p.requires_grad],
        lr=1e-4,
    )
    batch = {"frames": torch.randn(1, 4, 3, 256, 256)}
    metrics = trainer.train_step(batch, optimizer)
    assert isinstance(metrics, dict)
    assert "total_loss" in metrics
    assert isinstance(metrics["total_loss"], float)


def test_validation_step():
    trainer = _make_trainer()
    batch = {"frames": torch.randn(1, 4, 3, 256, 256)}
    metrics = trainer.validation_step(batch)
    assert isinstance(metrics, dict)
    assert "total_loss" in metrics
