#!/usr/bin/env python3
"""
Experiment 07: Temporal Compression Evaluation
Reproduces Table 5.

Compares all-intra vs temporal (IPPP, GOP=8) coding:
- Total BPP
- PSNR (dB)
- I-frame bits vs average P-frame bits
- P/I bit ratio
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from tabulate import tabulate

from common import (
    load_frames,
    encode_frames,
    decode_frames,
    compute_psnr,
    CHECKPOINT_M1,
    CHECKPOINT_M2,
    DATASET_DIR,
)

GOP = 8
LAMBDA_VAL = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SIZE = (256, 256)


def load_models(lambda_val: float, temporal: bool, device: str = DEVICE):
    from lewm_vc.encoder import LeWMEncoder
    from lewm_vc.working_decoder import LeWMDecoder
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer

    if temporal:
        from lewm_vc.predictor import LeWMPredictor

    encoder = LeWMEncoder(latent_dim=192, patch_size=16).to(device)
    decoder = LeWMDecoder(latent_dim=192).to(device)
    entropy = HyperpriorEntropy(latent_dim=192).to(device)
    quant = Quantizer(step_size=2.0 / 255.0)

    # Load intra-frame base
    ae_path = CHECKPOINT_M1 / f"ae_lambda_{lambda_val}_final.pt"
    entropy_path = CHECKPOINT_M1 / f"entropy_lambda_{lambda_val}_final.pt"

    if ae_path.exists():
        state = torch.load(ae_path, map_location=device, weights_only=True)
        encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in state.items() if "encoder" in k},
            strict=False,
        )
        decoder.load_state_dict(
            {k.replace("decoder.", ""): v for k, v in state.items() if "decoder" in k},
            strict=False,
        )
    if entropy_path.exists():
        entropy.load_state_dict(torch.load(entropy_path, map_location=device, weights_only=True))

    predictor = None
    if temporal:
        from lewm_vc.predictor import LeWMPredictor

        predictor = LeWMPredictor(latent_dim=192).to(device)
        temp_path = CHECKPOINT_M2 / "temporal_final.pt"
        if temp_path.exists():
            predictor.load_state_dict(torch.load(temp_path, map_location=device, weights_only=True))
        else:
            alt_path = CHECKPOINT_M2 / "temporal_epoch80.pt"
            if alt_path.exists():
                predictor.load_state_dict(
                    torch.load(alt_path, map_location=device, weights_only=True)
                )
            else:
                print("  [warn] No temporal checkpoint found, using random predictor")

    return encoder, decoder, entropy, quant, predictor


@torch.no_grad()
def evaluate_all_intra(frames: torch.Tensor, encoder, decoder, entropy, quant, device: str):
    """All-intra coding."""
    encoder.eval()
    decoder.eval()
    entropy.eval()
    latents = encoder(frames.to(device))
    qlatents = quant(latents)
    recon = decoder(qlatents, target_size=FRAME_SIZE).cpu()
    psnrs = [compute_psnr(recon[i : i + 1], frames[i : i + 1]) for i in range(frames.shape[0])]

    total_bits = sum(entropy(qlatents[i : i + 1]).sum().item() for i in range(qlatents.shape[0]))
    total_bits /= np.log(2)  # nats → bits
    bpp = total_bits / (frames.shape[0] * FRAME_SIZE[0] * FRAME_SIZE[1] * 3)
    return bpp, np.mean(psnrs), total_bits


@torch.no_grad()
def evaluate_temporal(
    frames: torch.Tensor, encoder, decoder, entropy, quant, predictor, device: str
):
    """Temporal IPPP coding with GOP=8."""
    encoder.eval()
    decoder.eval()
    entropy.eval()
    predictor.eval()
    T = frames.shape[0]
    i_bits_total = 0
    p_bits_total = 0
    n_p = 0
    context = []
    recon_frames = []
    psnrs = []

    for t in range(T):
        latent = encoder(frames[t : t + 1].to(device))
        if t % GOP == 0:
            # I-frame: code directly
            qlatent = quant(latent)
            i_bits_total += entropy(qlatent).sum().item()
            context = [qlatent.cpu()]
        else:
            # P-frame: predict then code residual
            context_tensor = torch.stack([c.to(device) for c in context[-4:]], dim=0).unsqueeze(1)
            mu, log_std = predictor(context_tensor)
            pred = mu[-1:]  # take last prediction
            residual = latent - pred.to(device)
            qres = quant(residual)
            p_bits_total += entropy(qres).sum().item()
            n_p += 1
            qlatent = pred.to(device) + qres
            context.append(qlatent.cpu())
            if len(context) > 4:
                context.pop(0)

        recon = decoder(qlatent, target_size=FRAME_SIZE).cpu()
        recon_frames.append(recon)
        psnrs.append(compute_psnr(recon, frames[t : t + 1]))

    total_bits_nats = i_bits_total + p_bits_total
    total_bits = total_bits_nats / np.log(2)
    bpp = total_bits / (T * FRAME_SIZE[0] * FRAME_SIZE[1] * 3)
    pi_ratio = (p_bits_total / n_p) / (i_bits_total / (T // GOP)) if n_p and T // GOP else 0

    return (
        bpp,
        np.mean(psnrs),
        total_bits_nats / np.log(2),
        i_bits_total / np.log(2),
        p_bits_total / (np.log(2) * max(n_p, 1)),
        pi_ratio,
    )


def main():
    print("=" * 70)
    print("Experiment 07: Temporal Compression Evaluation (Table 5)")
    print("=" * 70)

    pevid_files = sorted((DATASET_DIR / "pevid-hd").glob("*.mpg"))
    if not pevid_files:
        print("No PEViD-HD files. Run 01_download_data.sh first.")
        sys.exit(1)

    print(f"\nLoading test frames...")
    frames = load_frames(str(pevid_files[0]), max_frames=16)
    print(f"  Loaded {frames.shape[0]} frames")

    print(f"\n--- All-Intra Coding ---")
    enc, dec, ent, quant, _ = load_models(LAMBDA_VAL, temporal=False, device=DEVICE)
    bpp_i, psnr_i, bits_i_total = evaluate_all_intra(frames, enc, dec, ent, quant, DEVICE)
    print(f"  BPP = {bpp_i:.4f}, PSNR = {psnr_i:.2f} dB")

    print(f"\n--- Temporal Coding (IPPP, GOP={GOP}) ---")
    enc, dec, ent, quant, pred = load_models(LAMBDA_VAL, temporal=True, device=DEVICE)
    bpp_t, psnr_t, total_bits, i_bits, avg_p_bits, pi_ratio = evaluate_temporal(
        frames, enc, dec, ent, quant, pred, DEVICE
    )
    savings = (1 - bpp_t / bpp_i) * 100
    print(f"  BPP = {bpp_t:.4f}, PSNR = {psnr_t:.2f} dB")
    print(f"  I-frame bits = {i_bits:.0f}, Avg P-frame bits = {avg_p_bits:.0f}")
    print(f"  P/I ratio = {pi_ratio:.2f}x, Savings = {savings:.1f}%")

    print("\n" + "=" * 70)
    print("Table 5: Temporal Coding Results")
    print("=" * 70)
    rows = [
        ["All-intra", f"{bpp_i:.3f}", f"{psnr_i:.2f}", "--", "--"],
        ["Temporal", f"{bpp_t:.3f}", f"{psnr_t:.2f}", f"{i_bits:.0f}", f"{avg_p_bits:.0f}"],
    ]
    print(
        tabulate(
            rows,
            headers=["Mode", "BPP", "PSNR (dB)", "I-frame bits", "Avg P-frame bits"],
            tablefmt="grid",
        )
    )
    print(f"\n  Temporal savings: {savings:.1f}%")
    print(f"  P/I bit ratio: {pi_ratio:.2f}x")


if __name__ == "__main__":
    main()
