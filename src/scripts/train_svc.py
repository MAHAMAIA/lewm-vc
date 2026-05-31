"""
Train Dual-Layer SVC (Scalable Video Coding) for LeWM-VC.

Trains LatentSplitter and LatentFuser on top of a frozen Phase 1 checkpoint.
The result is a two-stream codec:
  - Base Layer (64ch, 4-bit): continuously streamed to cloud for AI pipeline
  - Enhancement Layer (128ch, 8-bit): stored locally on NVMe for forensic reconstruction

Usage:
    python -m src.scripts.train_svc --config configs/train_config.yaml \\
        --checkpoint checkpoints/lambda_0.05/final_step_100000.pt
    python -m src.scripts.train_svc --config configs/train_config.yaml \\
        --checkpoint checkpoints/lambda_0.05/final_step_100000.pt \\
        --lr 1e-3 --epochs 20
"""

import argparse
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


class SVCTrainer:
    """
    Trains the dual-layer SVC splitter and fuser on top of a frozen codec.

    Loss components:
      - L_bl: MSE between BL-only decode and original frame
      - L_full: MSE between BL+EL decode and original frame
      - L_rate: Bitrate penalty for BL (optional, encourages compression)
      - L_task: Task loss from frozen detection probe (optional)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        predictor: nn.Module,
        splitter: nn.Module,
        fuser: nn.Module,
        multi_quant: nn.Module,
        config: dict,
        device: str = "cuda",
        lambda_val: float = 0.05,
    ):
        self.device = device
        self.lambda_val = lambda_val

        # Frozen base modules
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

        # Trainable SVC modules
        self.splitter = splitter
        self.fuser = fuser
        self.multi_quant = multi_quant

        # SVCDecoder wraps frozen decoder + trainable fuser
        from lewm_vc import SVCDecoder as SVCWrapDecoder

        self.svc_decoder = SVCWrapDecoder(decoder, fuser=fuser).to(device)

        self.svc_decoder.base.requires_grad_(False)
        self.svc_decoder.base.eval()

        # Freeze base modules
        for m in [self.encoder, self.decoder, self.predictor]:
            m.requires_grad_(False)
            m.eval()

        # Config
        self.config = config
        svc_cfg = config.get("svc", {})
        self.bl_weight = svc_cfg.get("bl_loss_weight", 1.0)
        self.full_weight = svc_cfg.get("full_loss_weight", 1.0)
        self.task_weight = svc_cfg.get("task_loss_weight", 0.5)

        # TensorBoard
        self.writer = None
        log_cfg = config.get("logging", {})
        if HAS_TENSORBOARD and log_cfg.get("tensorboard", False):
            log_dir = Path(log_cfg.get("log_dir", "runs")) / f"svc_lambda_{lambda_val}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))

        # Checkpoint dir
        cp_cfg = config.get("checkpoint", {})
        self.checkpoint_dir = Path(cp_cfg.get("dir", "checkpoints")) / f"svc_lambda_{lambda_val}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    def _trainable_params(self) -> list[torch.Tensor]:
        """Get parameters that should be trained (splitter + fuser + multi_quant)."""
        params = []
        for m in [self.splitter, self.fuser, self.multi_quant]:
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def compute_loss(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with SVC split/quantize/decode, computing all loss components.

        Args:
            frames: [B, 3, H, W] input frames in [0, 1]

        Returns:
            Dict of loss components.
        """
        # Encode (frozen)
        latent = self.encoder(frames)

        # Split (trainable)
        bl, el = self.splitter(latent)

        # Quantize (trainable)
        bl_q, el_q = self.multi_quant(bl, el)

        # Decode both ways
        recon_bl = self.svc_decoder.decode_bl(bl_q)
        recon_full = self.svc_decoder.decode_full(bl_q, el_q)

        # Reconstruction losses
        mse = nn.functional.mse_loss
        loss_bl = mse(recon_bl, frames) * self.bl_weight
        loss_full = mse(recon_full, frames) * self.full_weight

        # Rate penalty for BL (encourage compact base layer)
        bl_bpp = bl_q.abs().mean()
        loss_rate = bl_bpp * 0.01 * self.lambda_val

        # Task loss (optional, 0 if probe not loaded)
        loss_task = torch.tensor(0.0, device=frames.device)
        if hasattr(self, "_probe") and self._probe is not None:
            loss_task = self._compute_task_loss(recon_bl, frames) * self.task_weight

        total = loss_bl + loss_full + loss_rate + loss_task

        return {
            "total_loss": total,
            "loss_bl": loss_bl,
            "loss_full": loss_full,
            "loss_rate": loss_rate,
            "loss_task": loss_task,
        }

    def _compute_task_loss(self, recon: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Task loss from a frozen detection probe."""
        if hasattr(self._probe, "compute_task_loss"):
            return self._probe.compute_task_loss(recon, original)
        feat_recon = self._probe(recon)
        feat_orig = self._probe(original)
        return nn.functional.mse_loss(feat_recon, feat_orig)

    def load_task_probe(self, probe: nn.Module | None = None):
        """
        Load a frozen feature extractor for task loss.
        If None, task loss is disabled.
        """
        if probe is not None:
            self._probe = probe.to(self.device)
            self._probe.requires_grad_(False)
            self._probe.eval()
            print(f"  Task probe loaded: {type(probe).__name__}")
        else:
            self._probe = None
            print("  Task loss disabled")

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """Single training step."""
        frames = batch["frames"].to(self.device)
        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w)

        losses = self.compute_loss(frames)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validation_step(self, batch: dict) -> dict[str, float]:
        """Validation step."""
        frames = batch["frames"].to(self.device)
        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            frames = frames.view(b * t, c, h, w)
        losses = self.compute_loss(frames)
        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, name: str, optimizer: torch.optim.Optimizer | None = None) -> str:
        """Save SVC checkpoint (splitter + fuser + multi_quant only)."""
        path = self.checkpoint_dir / f"{name}.pt"
        ckpt = {
            "global_step": self.global_step,
            "lambda_val": self.lambda_val,
            "splitter": self.splitter.state_dict(),
            "fuser": self.fuser.state_dict(),
            "multi_quant": self.multi_quant.state_dict(),
        }
        if optimizer is not None:
            ckpt["optimizer"] = optimizer.state_dict()
        torch.save(ckpt, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load SVC checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.splitter.load_state_dict(ckpt["splitter"])
        self.fuser.load_state_dict(ckpt["fuser"])
        self.multi_quant.load_state_dict(ckpt["multi_quant"])
        self.global_step = ckpt.get("global_step", 0)
        self.lambda_val = ckpt.get("lambda_val", self.lambda_val)
        print(f"  Resumed SVC from {checkpoint_path} (step {self.global_step})")

    def log_metrics(self, metrics: dict[str, float], step: int):
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"svc/{name}", value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def load_base_checkpoint(
    checkpoint_path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    predictor: nn.Module,
    device: str,
) -> dict:
    """Load a Phase 1 checkpoint and return configuration."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    models_state = ckpt.get("models", ckpt)

    def _load_state(model, key: str):
        if key in models_state:
            try:
                model.load_state_dict(models_state[key])
                print(f"  Loaded {key}")
            except Exception as e:
                print(f"  [warn] {key}: {e}")

    _load_state(encoder, "encoder")
    _load_state(decoder, "decoder")
    _load_state(predictor, "predictor")

    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Layer SVC")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Phase 1 checkpoint to load frozen base models from",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_val",
        default=0.05,
        help="RD lambda (for checkpoint naming)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for SVC modules")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from SVC checkpoint")
    parser.add_argument(
        "--task-probe",
        type=str,
        default=None,
        help="Backbone for task probe: resnet18, resnet34, resnet50 (default: none)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create base models
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.svc import LatentFuser, LatentSplitter, MultiRateQuantizer

    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)
    svc_cfg = config.get("svc", {})
    base_dim = svc_cfg.get("base_dim", 64)

    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=model_cfg.get("patch_size", 16),
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 192),
        num_layers=model_cfg.get("encoder", {}).get("num_layers", 6),
        num_heads=model_cfg.get("encoder", {}).get("num_heads", 3),
        semantic_surprise=False,
    )
    decoder = LeWMDecoder(latent_dim=latent_dim)
    predictor = LeWMPredictor(
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("predictor", {}).get("hidden_dim", 256),
        num_layers=model_cfg.get("predictor", {}).get("num_layers", 8),
        num_heads=model_cfg.get("predictor", {}).get("num_heads", 4),
    )
    for m in [encoder, decoder, predictor]:
        m.to(device)

    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint: {args.checkpoint}")
    load_base_checkpoint(args.checkpoint, encoder, decoder, predictor, device)

    # Create SVC modules (with learned projections)
    print("\nCreating SVC modules...")
    splitter = LatentSplitter(
        latent_dim=latent_dim,
        base_dim=base_dim,
        use_learned_split=True,
    ).to(device)
    fuser = LatentFuser(
        latent_dim=latent_dim,
        base_dim=base_dim,
        use_learned_fusion=True,
    ).to(device)
    multi_quant = MultiRateQuantizer(
        num_levels_bl=2 ** svc_cfg.get("bl_quant_bits", 4),
        num_levels_el=2 ** svc_cfg.get("el_quant_bits", 8),
    ).to(device)

    n_svc_params = (
        sum(p.numel() for p in splitter.parameters())
        + sum(p.numel() for p in fuser.parameters())
        + sum(p.numel() for p in multi_quant.parameters())
    )
    print(f"  SVC trainable params: {n_svc_params:,}")

    # Initialize trainer
    trainer = SVCTrainer(
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        splitter=splitter,
        fuser=fuser,
        multi_quant=multi_quant,
        config=config,
        device=device,
        lambda_val=args.lambda_val,
    )

    # Resume from SVC checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Load task probe
    if args.task_probe:
        from lewm_vc.utils.task_probe import create_task_probe

        try:
            probe = create_task_probe(args.task_probe, multi_scale=True, device=device)
            trainer.load_task_probe(probe)
        except Exception as e:
            print(f"  [warn] Task probe failed to load: {e}")
            trainer.load_task_probe(None)
    else:
        trainer.load_task_probe(None)

    # Create dataset
    print("\nCreating dataset...")
    from lewm_vc.data import FrameDataset, collate_sequences

    data_cfg = config.get("data", {})
    roots = data_cfg.get("roots", [])
    image_size = data_cfg.get("image_size", 256)

    train_ds = FrameDataset.from_roots(
        roots,
        sequence_length=1,
        image_size=image_size,
        augment=True,
        split="train",
    )
    val_ds = FrameDataset.from_roots(
        roots,
        sequence_length=1,
        image_size=image_size,
        augment=False,
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

    # Optimizer (SVC modules only)
    optimizer = torch.optim.AdamW(
        trainer._trainable_params(),
        lr=args.lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log_interval = train_cfg.get("log_interval", 10)
    val_interval = train_cfg.get("val_interval", 500)
    save_interval = train_cfg.get("save_interval", 5000)

    print(f"\n{'=' * 50}")
    print(f"SVC Training — {args.epochs} epochs, lr={args.lr}")
    print(f"{'=' * 50}")

    try:
        for epoch in range(1, args.epochs + 1):
            trainer.splitter.train()
            trainer.fuser.train()
            trainer.multi_quant.train()

            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                train_metrics = trainer.train_step(batch, optimizer)
                epoch_loss += train_metrics["total_loss"]
                num_batches += 1

                if trainer.global_step % log_interval == 0:
                    trainer.log_metrics(train_metrics, trainer.global_step)

                if trainer.global_step % val_interval == 0:
                    val_batch = next(iter(val_loader))
                    val_metrics = trainer.validation_step(val_batch)
                    trainer.log_metrics(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        trainer.global_step,
                    )
                    print(
                        f"  epoch={epoch} step={trainer.global_step} "
                        f"loss={train_metrics['total_loss']:.4f} "
                        f"bl={train_metrics['loss_bl']:.4f} "
                        f"full={train_metrics['loss_full']:.4f} "
                        f"val_loss={val_metrics['total_loss']:.4f}"
                    )

                if trainer.global_step % save_interval == 0:
                    trainer.save_checkpoint(f"svc_step_{trainer.global_step}", optimizer)

            scheduler.step()
            avg_loss = epoch_loss / max(1, num_batches)
            print(
                f"Epoch {epoch}/{args.epochs} — avg loss: {avg_loss:.4f}, "
                f"lr: {scheduler.get_last_lr()[0]:.2e}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint...")
    finally:
        trainer.save_checkpoint(f"svc_final_step_{trainer.global_step}", optimizer)
        trainer.close()

    print(f"\nDone. SVC checkpoint: {trainer.checkpoint_dir}")
    print(f"Save path: {trainer.checkpoint_dir / f'svc_final_step_{trainer.global_step}.pt'}")


if __name__ == "__main__":
    main()
