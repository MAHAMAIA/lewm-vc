"""
Training Pipeline for LeWM-VC Video Codec

Phases (configurable via train_config.yaml):
  Phase 0: JEPA warmup — train encoder + predictor, no RD
  Phase 1: Joint RD — full loss with rate-distortion
  Phase 2: QAT — quantization-aware training
  Phase 3: Decoder refinement — adapt decoder to QAT-hardened latents
  Phase 4: Cooldown — high perceptual weight, lower LR

Loss formula (paper eq. 5):
    L_Total = R + lambda*D + gamma*L_JEPA + delta*L_SIGReg
  where L_JEPA = ||g_phi(z_{<t}) - z_t||^2_2
  and   L_SIGReg = Cramér-Wold random-projection SIGReg (LeJEPA)

Usage:
    python -m src.scripts.train --config configs/train_config.yaml
    python -m src.scripts.train --config configs/train_config.yaml --lambda 0.05
    python -m src.scripts.train --config configs/train_config.yaml --resume checkpoints/lambda_0.05/step_10000.pt
"""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from lewm_vc.utils import SIGReg

# Will be imported in __main__ after path setup
# from lewm_vc.data import FrameDataset, collate_sequences


class TrainingPhase:
    DECODER_WARMUP = 0
    JOINT_RD = 1
    QAT = 2
    DISTILLATION = 3
    COOLDOWN = 4

    @staticmethod
    def get_name(phase: int) -> str:
        return {0: "warmup", 1: "joint_rd", 2: "qat", 3: "distillation", 4: "cooldown"}.get(
            phase, "unknown"
        )


class LeWMTrainer:
    """
    Training loop for LeWM-VC with mixed precision and configurable phases.

    Args:
        lambda_val: RD trade-off for this training run (e.g. 0.05).
        resume_from: Path to checkpoint to resume from.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        entropy_model: nn.Module,
        quantizer: nn.Module,
        rate_controller: nn.Module,
        config: dict,
        device: str = "cuda",
        lambda_val: float = 0.05,
        resume_from: str | None = None,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.entropy_model = entropy_model
        self.quantizer = quantizer
        self.rate_controller = rate_controller
        self.config = config
        self.device = device
        self.lambda_val = lambda_val

        self.models = {
            "encoder": encoder,
            "predictor": predictor,
            "decoder": decoder,
            "entropy_model": entropy_model,
            "quantizer": quantizer,
            "rate_controller": rate_controller,
        }

        # TensorBoard
        self.writer = None
        log_cfg = config.get("logging", {})
        if HAS_TENSORBOARD and log_cfg.get("tensorboard", False):
            log_dir = Path(log_cfg.get("log_dir", "runs")) / f"lambda_{lambda_val}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))

        # Checkpoint dir
        cp_cfg = config.get("checkpoint", {})
        self.checkpoint_dir = Path(cp_cfg.get("dir", "checkpoints")) / f"lambda_{lambda_val}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = cp_cfg.get("keep_last", 5)
        self._checkpoint_paths: list[Path] = []

        # SIGReg (Cramér-Wold random-projection regularizer)
        self.sigreg = SIGReg()

        # Mixed precision
        prec = config.get("training", {}).get("precision", "no")
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}.get(prec, None)
        self.amp_enabled = self.amp_dtype is not None

        # Phase state
        self.current_phase = 0
        self.global_step = 0
        self.phase_step = 0

        # Load checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            self._setup_phase(0)

    def _phase_config(self, phase: int | None = None) -> dict:
        """Get config dict for a given phase index."""
        phase = phase if phase is not None else self.current_phase
        key = f"phase{phase}"
        phases = self.config.get("phases", {})
        return phases.get(key, {})

    def _setup_phase(self, phase: int):
        """Freeze/unfreeze models per phase config."""
        self.current_phase = phase
        self.phase_step = 0
        phase_cfg = self._phase_config(phase)
        freeze_list = phase_cfg.get("freeze", [])

        # Set quantizer mode
        if phase == TrainingPhase.QAT:
            self.quantizer.set_mode("inference")

        # Freeze specified modules
        for name, model in self.models.items():
            if name in freeze_list:
                for p in model.parameters():
                    p.requires_grad = False
            else:
                for p in model.parameters():
                    p.requires_grad = True

        name = TrainingPhase.get_name(phase)
        frozen = [n for n in freeze_list if n in self.models]
        active = [n for n in self.models if n not in freeze_list]
        print(f"  Phase {phase} ({name}): active={active}, frozen={frozen}")

    def compute_loss(
        self,
        frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss per paper eq. 5.

        Args:
            frames: [B, T, 3, H, W] in [0, 1]

        Returns:
            Dict of loss components.
        """
        phase_cfg = self._phase_config()
        gamma = phase_cfg.get("gamma", 1.0)
        delta = phase_cfg.get("delta", 0.01)
        lambda_val = phase_cfg.get("lambda", self.lambda_val)

        b, num_frames = frames.shape[:2]

        latents = []
        recon_latents = []
        surprises = []
        jepa_loss = torch.tensor(0.0, device=frames.device)
        sigreg_loss = torch.tensor(0.0, device=frames.device)
        rates = []
        reconstructions = []

        for frame_idx in range(num_frames):
            frame = frames[:, frame_idx]
            encoder_output = self.encoder(frame, return_surprise=True)
            if isinstance(encoder_output, tuple):
                latent, surprise = encoder_output
            else:
                latent, surprise = encoder_output, None
            latents.append(latent)
            if surprise is not None:
                surprises.append(surprise)

            if frame_idx == 0:
                # I-frame: quantize raw latent, decode, store as context
                quant_latent = self.quantizer(latent)
                recon = self.decoder(quant_latent)
                reconstructions.append(recon)
                recon_latents.append(quant_latent.detach())
            else:
                # P-frame: predict from decoded latents, code residual,
                #          reconstruct latent for decoding and as future context
                ctx = recon_latents[max(0, len(recon_latents) - self.predictor.context_len) :]
                pred_mean, pred_std = self.predictor(ctx)
                # JEPA loss on raw encoder output (stop-gradient target)
                jepa_loss = jepa_loss + nn.functional.mse_loss(pred_mean, latent.detach())

                residual = latent - pred_mean
                quant_residual = self.quantizer(residual)
                rate, _ = self.entropy_model(quant_residual)
                rates.append(rate)

                recon_latent = pred_mean + quant_residual
                recon = self.decoder(recon_latent)
                reconstructions.append(recon)
                recon_latents.append(recon_latent.detach())

        reconstructions = torch.stack(reconstructions, dim=1)

        # SIGReg (Cramér-Wold random projections) on all latents
        latent_stack = torch.stack(latents, dim=0)  # (T, B, D, H, W)
        latent_pooled = latent_stack.mean(dim=(-1, -2))  # (T, B, D) — spatial avg
        sigreg_loss = self.sigreg(latent_pooled)

        mse_loss = nn.functional.mse_loss(reconstructions, frames)
        lpips_loss = self._compute_lpips_loss(reconstructions, frames)
        mse_w = phase_cfg.get("mse_weight", 0.7)
        lpips_w = phase_cfg.get("lpips_weight", 0.3)
        distortion_loss = mse_w * mse_loss + lpips_w * lpips_loss

        total_rate = torch.stack(rates).sum() if rates else torch.tensor(0.0, device=frames.device)
        rate_loss = total_rate.clamp(max=1e12)

        surprise_loss = torch.tensor(0.0, device=frames.device)
        if surprises:
            surprise_loss = 0.01 * torch.stack([s.mean() for s in surprises]).mean()

        total_loss = (
            rate_loss
            + lambda_val * distortion_loss
            + gamma * jepa_loss
            + delta * sigreg_loss
            + surprise_loss
        )
        total_loss = torch.nan_to_num(total_loss, nan=1e5, posinf=1e5, neginf=1e5)

        return {
            "total_loss": total_loss,
            "rate_loss": rate_loss,
            "distortion_loss": distortion_loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
            "jepa_loss": jepa_loss,
            "sigreg_loss": sigreg_loss,
            "surprise_loss": surprise_loss,
        }

    def _compute_lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss.

        Uses the 'lpips' package if available; falls back to MSE.
        """
        if getattr(self, "_lpips_fn", None) is not None:
            b, t = pred.shape[:2]
            # LPIPS expects [B, 3, H, W], reshape batch+time
            p = pred.view(b * t, *pred.shape[2:])
            t_ = target.view(b * t, *target.shape[2:])
            return self._lpips_fn(p, t_).mean()
        return nn.functional.mse_loss(pred, target)

    def _init_lpips(self):
        """Try to initialize LPIPS; silently fall back to MSE."""
        try:
            import lpips

            self._lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
            self._lpips_fn.eval()
            for p in self._lpips_fn.parameters():
                p.requires_grad = False
            print("  LPIPS perceptual loss enabled")
        except ImportError:
            self._lpips_fn = None
            print("  LPIPS not installed, using MSE for perceptual loss")

    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None = None,
    ) -> dict[str, float]:
        """Single training step with optional mixed precision."""
        frames = batch["frames"].to(self.device)

        if self.amp_enabled:
            with torch.amp.autocast(device_type=self.device, dtype=self.amp_dtype):
                losses = self.compute_loss(frames)
        else:
            losses = self.compute_loss(frames)

        if scaler is not None:
            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            # Clean NaN gradients before clipping (prevents optimizer.step() from corrupting weights)
            for p in [
                m for ms in self.models.values() for m in ms.parameters() if m.grad is not None
            ]:
                p.grad.nan_to_num_()
            torch.nn.utils.clip_grad_norm_(
                [p for m in self.models.values() for p in m.parameters() if p.requires_grad],
                max_norm=self.config.get("training", {}).get("max_grad_norm", 1.0),
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total_loss"].backward()
            for p in [
                m for ms in self.models.values() for m in ms.parameters() if m.grad is not None
            ]:
                p.grad.nan_to_num_()
            torch.nn.utils.clip_grad_norm_(
                [p for m in self.models.values() for p in m.parameters() if p.requires_grad],
                max_norm=self.config.get("training", {}).get("max_grad_norm", 1.0),
            )
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict[str, float]:
        """Single validation step."""
        frames = batch["frames"].to(self.device)
        losses = self.compute_loss(frames)
        return {k: v.item() for k, v in losses.items()}

    def switch_phase(self, new_phase: int):
        """Advance to next training phase."""
        old_name = TrainingPhase.get_name(self.current_phase)
        new_name = TrainingPhase.get_name(new_phase)
        print(f"  Phase transition: {old_name} -> {new_name}")
        self._setup_phase(new_phase)

    def save_checkpoint(self, name: str, optimizer: torch.optim.Optimizer | None = None) -> str:
        """Save checkpoint with model + optimizer state."""
        path = self.checkpoint_dir / f"{name}.pt"

        ckpt = {
            "phase": self.current_phase,
            "global_step": self.global_step,
            "phase_step": self.phase_step,
            "lambda_val": self.lambda_val,
            "config": self.config,
            "models": {},
        }
        for m_name, model in self.models.items():
            ckpt["models"][m_name] = model.state_dict()

        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()

        torch.save(ckpt, path)

        # Prune old checkpoints
        self._checkpoint_paths.append(path)
        if len(self._checkpoint_paths) > self.keep_last:
            old = self._checkpoint_paths.pop(0)
            if old.exists():
                old.unlink()

        return str(path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore phase/step state."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        for name, state_dict in ckpt["models"].items():
            if name in self.models:
                try:
                    self.models[name].load_state_dict(state_dict)
                except Exception as e:
                    print(f"  [warning] failed to load {name}: {e}")

        # Zero-init entropy model's final layer (checkpoint overwrites it with random weights)
        nn.init.zeros_(self.entropy_model.hyperprior_cnn[-1].weight)
        nn.init.zeros_(self.entropy_model.hyperprior_cnn[-1].bias)

        self.current_phase = ckpt.get("phase", 0)
        self._setup_phase(self.current_phase)
        self.global_step = ckpt.get("global_step", 0)
        self.phase_step = ckpt.get("phase_step", 0)
        self.lambda_val = ckpt.get("lambda_val", self.lambda_val)

        # Get optimizer state if available
        self._resume_optimizer = ckpt.get("optimizer")

        print(f"  Resumed from {checkpoint_path}")
        print(
            f"  Phase={self.current_phase}, global_step={self.global_step}, phase_step={self.phase_step}"
        )

    def log_metrics(self, metrics: dict[str, float], step: int):
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def train(
    encoder: nn.Module,
    predictor: nn.Module,
    decoder: nn.Module,
    entropy_model: nn.Module,
    quantizer: nn.Module,
    rate_controller: nn.Module,
    config: dict,
    device: str = "cuda",
    lambda_val: float = 0.05,
    resume_from: str | None = None,
    start_phase: int = 0,
) -> None:
    """Main training loop."""
    from lewm_vc.data import FrameDataset, collate_sequences

    # Init trainer
    trainer = LeWMTrainer(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device=device,
        lambda_val=lambda_val,
        resume_from=resume_from,
    )

    # Init LPIPS
    trainer._init_lpips()

    # Create datasets
    data_cfg = config.get("data", {})
    roots = data_cfg.get("roots", [])
    if not roots:
        print("Error: no data roots configured in data.roots")
        return

    seq_len = data_cfg.get("sequence_length", 16)
    image_size = data_cfg.get("image_size", 256)
    frame_stride = data_cfg.get("frame_stride", 1)
    augment = data_cfg.get("augment", False)
    print(f"\nCreating datasets from {len(roots)} root(s)...")
    for r in roots:
        print(f"  {r}")

    train_ds = FrameDataset.from_roots(
        roots,
        sequence_length=seq_len,
        image_size=image_size,
        augment=augment,
        frame_stride=frame_stride,
        split="train",
    )
    val_ds = FrameDataset.from_roots(
        roots,
        sequence_length=seq_len,
        image_size=image_size,
        augment=False,
        frame_stride=1,
        split="val",
    )

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        collate_fn=collate_sequences,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sequences,
    )

    # Optimizer with per-module LR
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": train_cfg["lr_encoder"]},
            {"params": predictor.parameters(), "lr": train_cfg["lr_predictor"]},
            {"params": decoder.parameters(), "lr": train_cfg["lr_decoder"]},
            {"params": entropy_model.parameters(), "lr": train_cfg["lr_entropy"]},
            {"params": rate_controller.parameters(), "lr": train_cfg.get("lr_rate_control", 1e-4)},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Restore optimizer state if resumed
    if (
        resume_from
        and hasattr(trainer, "_resume_optimizer")
        and trainer._resume_optimizer is not None
    ):
        optimizer.load_state_dict(trainer._resume_optimizer)
        print("  Optimizer state restored")

    # LR scheduler
    lr_scheduler_type = train_cfg.get("lr_scheduler", "cosine")
    warmup_steps = train_cfg.get("lr_warmup_steps", 1000)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if lr_scheduler_type == "cosine":
            total = sum(
                config.get("phases", {}).get(f"phase{p}", {}).get("steps", 0) for p in range(5)
            )
            progress = (step - warmup_steps) / max(1, total - warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Mixed precision scaler
    precision = train_cfg.get("precision", "no")
    scaler = None
    if precision == "fp16":
        scaler = torch.amp.GradScaler(device=device)
    elif precision == "bf16":
        scaler = None  # bf16 doesn't need gradient scaling

    # Phase durations
    phase_durations = {}
    for p in range(5):
        phase_durations[p] = config.get("phases", {}).get(f"phase{p}", {}).get("steps", 0)

    # Logging intervals
    log_interval = train_cfg.get("log_interval", 10)
    val_interval = train_cfg.get("val_interval", 500)
    save_interval = train_cfg.get("save_interval", 5000)

    # Skip phases before start_phase
    for p in range(start_phase):
        trainer.switch_phase(p)
        trainer.current_phase = p
        trainer.phase_step = phase_durations[p]
        trainer.switch_phase(p + 1)

    # Training loop
    try:
        for phase in range(start_phase, 5):
            if trainer.current_phase != phase:
                trainer.switch_phase(phase)
            target_steps = phase_durations[phase]

            if target_steps <= 0:
                continue

            print(f"\n{'=' * 50}")
            print(f"Phase {phase} ({TrainingPhase.get_name(phase)}) — {target_steps} steps")
            print(f"{'=' * 50}")

            phase_cfg = config.get("phases", {}).get(f"phase{phase}", {})
            entropy_warmup = phase_cfg.get("entropy_warmup_steps", 0)

            epoch = 0
            while trainer.phase_step < target_steps:
                epoch += 1
                for batch in train_loader:
                    if trainer.phase_step >= target_steps:
                        break

                    # Entropy warmup: freeze extra modules so rate gradient
                    # trains only entropy model on stationary residuals
                    if entropy_warmup > 0 and trainer.phase_step < entropy_warmup:
                        for name in ["predictor", "decoder"]:
                            for p in trainer.models[name].parameters():
                                p.requires_grad = False

                    train_metrics = trainer.train_step(batch, optimizer, scaler)
                    scheduler.step()
                    trainer.global_step += 1
                    trainer.phase_step += 1

                    # Restore requires_grad after warmup step
                    if entropy_warmup > 0 and trainer.phase_step <= entropy_warmup:
                        for name in ["predictor", "decoder"]:
                            for p in trainer.models[name].parameters():
                                p.requires_grad = True

                    # Logging
                    if trainer.global_step % log_interval == 0:
                        trainer.log_metrics(
                            {f"train/{k}": v for k, v in train_metrics.items()},
                            trainer.global_step,
                        )

                    # Validation (skip if val set is empty)
                    if trainer.global_step % val_interval == 0:
                        val_iter = iter(val_loader)
                        val_batch = next(val_iter, None)
                        if val_batch is not None:
                            val_metrics = trainer.validation_step(val_batch)
                            trainer.log_metrics(
                                {f"val/{k}": v for k, v in val_metrics.items()},
                                trainer.global_step,
                            )
                        print(
                            f"  step={trainer.global_step} "
                            f"loss={train_metrics['total_loss']:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.2e}"
                        )

                    # Save checkpoint
                    if trainer.global_step % save_interval == 0:
                        path = trainer.save_checkpoint(f"step_{trainer.global_step}", optimizer)
                        print(f"  Saved: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint...")
        trainer.save_checkpoint(f"interrupt_step_{trainer.global_step}", optimizer)
    finally:
        # Save final checkpoint
        trainer.save_checkpoint(f"final_step_{trainer.global_step}", optimizer)
        trainer.close()

    print(f"\nTraining complete. Final step: {trainer.global_step}")
    print(f"Checkpoints: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LeWM-VC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu)")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_val",
        default=0.05,
        help="RD lambda value for this training run",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Start from this phase (for resume)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Import model classes
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils import RateController

    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)

    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=model_cfg.get("patch_size", 16),
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 192),
        num_layers=model_cfg.get("encoder", {}).get("num_layers", 6),
        num_heads=model_cfg.get("encoder", {}).get("num_heads", 3),
        semantic_surprise=model_cfg.get("encoder", {}).get("semantic_surprise", True),
    )
    predictor = LeWMPredictor(
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("predictor", {}).get("hidden_dim", 256),
        num_layers=model_cfg.get("predictor", {}).get("num_layers", 8),
        num_heads=model_cfg.get("predictor", {}).get("num_heads", 4),
        context_len=model_cfg.get("predictor", {}).get("context_len", 4),
    )
    decoder = LeWMDecoder(
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("decoder", {}).get("hidden_dim", 512),
    )
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=model_cfg.get("entropy", {}).get("hyper_channels", 256),
        num_components=model_cfg.get("entropy", {}).get("num_components", 2),
    )
    quantizer = Quantizer()
    rate_controller = RateController(latent_dim=latent_dim)

    # Move to device
    for m in [encoder, predictor, decoder, entropy_model, quantizer, rate_controller]:
        m.to(args.device)

    train(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        entropy_model=entropy_model,
        quantizer=quantizer,
        rate_controller=rate_controller,
        config=config,
        device=args.device,
        lambda_val=args.lambda_val,
        resume_from=args.resume,
        start_phase=args.phase,
    )
