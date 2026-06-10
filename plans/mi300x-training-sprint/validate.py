"""
Step 4: Validation Suite — FPN Compressor Evaluation
Modes:
  recon  — feature-space metrics (PSNR, MSE, BPP) on held-out features
  map    — downstream detection mAP on COCO val2017 via Faster R-CNN
  probe  — latent probe accuracy (train CNN on latents to predict detections)
  all    — run all three
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ======================================================================
# Mode 1: Feature-space reconstruction metrics
# ======================================================================


class FPNFeatureDataset(Dataset):
    def __init__(self, feature_dir: str, level: str, split: str):
        self.files = sorted(Path(feature_dir) / level / split).glob("feat_*.pt")
        self.files = list(self.files)
        if not self.files:
            raise FileNotFoundError(f"{feature_dir}/{level}/{split}: no .pt files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        return {"features": data["features"], "path": str(self.files[idx])}


def collate_feats(batch):
    return {"features": torch.stack([b["features"] for b in batch])}


def nll_rate_loss(latents, mu, sigma, step_size):
    half = step_size / 2.0
    prob = torch.special.ndtr((latents + half - mu) / sigma) - torch.special.ndtr(
        (latents - half - mu) / sigma
    )
    prob = prob.clamp(min=1e-10)
    return (-torch.log(prob) / 0.693147).sum() / latents.shape[0]


@torch.no_grad()
def eval_recon(compressor, decompressor, entropy, quantizer, loader, step_size, device):
    compressor.eval()
    decompressor.eval()
    entropy.eval()
    total_mse, total_bpp, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["features"].to(device)
        B, C, H, W = x.shape
        latent = compressor(x)
        qz = quantizer(latent)
        recon = decompressor(qz)
        mse = F.mse_loss(recon, x, reduction="sum").item()
        _, params = entropy(latent)
        rate = nll_rate_loss(latent, params["mu"], params["sigma"], step_size)
        total_mse += mse
        total_bpp += rate / (H * W) * B
        n += B * C * H * W
    avg_mse = total_mse / n
    psnr_val = 10 * torch.log10(torch.tensor(1.0 / max(avg_mse, 1e-10))).item()
    return {"psnr": psnr_val, "mse": avg_mse, "bpp": total_bpp / (n // (C * H * W))}


def run_recon(args):
    from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer, QuantMode

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ld = ckpt.get("latent_dim", 16)
    hc = ckpt.get("hyper_channels", 32)
    level = ckpt.get("fpn_level", args.fpn_level)

    comp = FeatureCompressor(in_channels=256, latent_dim=ld).to(device).eval()
    decomp = FeatureDecompressor(latent_dim=ld, out_channels=256).to(device).eval()
    ent = HyperpriorEntropy(latent_dim=ld, hyper_channels=hc).to(device).eval()
    comp.load_state_dict(ckpt["compressor"])
    decomp.load_state_dict(ckpt["decompressor"])
    ent.load_state_dict(ckpt["entropy"])

    quant = Quantizer(mode=QuantMode.INFERENCE).to(device)
    step_size = quant.step_size.item()

    ds = FPNFeatureDataset(args.feature_dir, level, "test")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=collate_feats)
    metrics = eval_recon(comp, decomp, ent, quant, loader, step_size, device)
    print(
        f"\n  [recon] {level}: PSNR={metrics['psnr']:.2f} dB, "
        f"MSE={metrics['mse']:.6f}, BPP={metrics['bpp']:.4f}"
    )
    return metrics


# ======================================================================
# Mode 2: Downstream detection mAP
# ======================================================================


def run_map(args):
    from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor
    from lewm_vc.quant import Quantizer, QuantMode

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ld = ckpt.get("latent_dim", 16)
    level = ckpt.get("fpn_level", args.fpn_level)
    comp = FeatureCompressor(in_channels=256, latent_dim=ld).to(device).eval()
    decomp = FeatureDecompressor(latent_dim=ld, out_channels=256).to(device).eval()
    comp.load_state_dict(ckpt["compressor"])
    decomp.load_state_dict(ckpt["decompressor"])
    quant = Quantizer(mode=QuantMode.INFERENCE).to(device)

    # Build detector
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    detector = (
        fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
    )
    for p in detector.parameters():
        p.requires_grad = False

    # Map level name to FPN key index
    fpn_key_map = {"P2": "0", "P3": "1", "P4": "2", "P5": "3"}
    fpn_key = fpn_key_map.get(level, "2")

    # Setup COCO
    coco_dir = Path(args.coco_dir)
    val_dir = coco_dir / "val2017"
    ann_file = coco_dir / "annotations/instances_val2017.json"

    if not val_dir.exists():
        print(f"  COCO val2017 not found at {val_dir}. Download with:")
        print(f"  wget http://images.cocodataset.org/zips/val2017.zip -P /tmp/")
        print(f"  unzip /tmp/val2017.zip -d {coco_dir}")
        return None

    from pycocotools.coco import COCO
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    import torchvision.transforms as T

    coco_gt = COCO(str(ann_file))
    all_ids = coco_gt.getImgIds()
    random.seed(42)
    random.shuffle(all_ids)
    val_ids = all_ids[: args.num_images]

    # Build target cache
    targets = {}
    for img_id in val_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        targets[img_id] = {
            "boxes": torch.tensor(
                [
                    [
                        a["bbox"][0],
                        a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3],
                    ]
                    for a in anns
                ]
            ),
            "labels": torch.tensor([a["category_id"] for a in anns]),
        }

    def detect_original(img):
        return detector([img])[0]

    @torch.no_grad()
    def detect_compressed(img):
        img_list, _ = detector.transform([img])
        x = img_list.tensors
        fpn_feats = detector.backbone(x)
        compressed = comp(fpn_feats[fpn_key])
        qz = quant(compressed)
        reconstructed = decomp(qz)
        fpn_feats[fpn_key] = reconstructed
        proposals = detector.rpn(img_list, fpn_feats)[0]
        dets = detector.roi_heads(fpn_feats, proposals, img_list.image_sizes)[0]
        dets = detector.transform.postprocess([dets], img_list.image_sizes, [tuple(img.shape[-2:])])
        return dets[0]

    to_tensor = T.Compose([T.ToTensor()])

    def evaluate(det_fn, name):
        metric = MeanAveragePrecision(iou_type="bbox")
        preds_list, targets_list = [], []
        for img_id in tqdm(val_ids, desc=name):
            fpath = val_dir / f"{img_id:012d}.jpg"
            if not fpath.exists():
                continue
            img = to_tensor(Image.open(fpath).convert("RGB")).to(device)
            det = det_fn(img)
            anns = [a for a in coco_gt.dataset["annotations"] if a["image_id"] == img_id]
            if not anns:
                continue
            preds_list.append(
                {
                    "boxes": det["boxes"].cpu(),
                    "scores": det["scores"].cpu(),
                    "labels": det["labels"].cpu(),
                }
            )
            targets_list.append(targets[img_id])
        metric.update(preds_list, targets_list)
        r = metric.compute()
        print(f"  {name}: mAP@0.50:0.95={r['map']:.4f}")
        return r

    from PIL import Image

    print(f"\n  [map] Evaluating {level} compression on COCO val2017 ({len(val_ids)} images)...")
    t0 = time.time()
    orig = evaluate(detect_original, "Original")
    comp_r = evaluate(detect_compressed, f"Compressed ({level})")
    elapsed = time.time() - t0

    delta = float(orig["map"]) - float(comp_r["map"])
    delta_pct = (delta / float(orig["map"])) * 100 if float(orig["map"]) > 0 else 0
    verdict = "PASS (<10%)" if delta_pct < 10 else "FAIL (>10%)"
    if delta_pct < 5:
        verdict = "EXCELLENT (<5%)"

    print(f"\n  {'=' * 50}")
    print(f"  mAP Results [{elapsed:.0f}s]")
    print(f"  {'=' * 50}")
    print(f"  Original:              mAP@0.50:0.95 = {orig['map']:.4f}")
    print(f"  Compressed ({level}):       mAP@0.50:0.95 = {comp_r['map']:.4f}")
    print(f"  Delta:                 {delta:.4f} ({delta_pct:.1f}%)")
    print(f"  Verdict:               {verdict}")

    return {
        "level": level,
        "original_map": float(orig["map"]),
        "compressed_map": float(comp_r["map"]),
        "delta": delta,
        "delta_pct": delta_pct,
        "verdict": verdict,
    }


# ======================================================================
# Mode 3: Latent probe
# ======================================================================


class LatentProbe(nn.Module):
    """Lightweight CNN probe trained on latents to predict objectness."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_probe(args):
    from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor
    from lewm_vc.quant import Quantizer, QuantMode

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ld = ckpt.get("latent_dim", 16)
    level = ckpt.get("fpn_level", args.fpn_level)
    comp = FeatureCompressor(in_channels=256, latent_dim=ld).to(device).eval()
    decomp = FeatureDecompressor(latent_dim=ld, out_channels=256).to(device).eval()
    comp.load_state_dict(ckpt["compressor"])
    decomp.load_state_dict(ckpt["decompressor"])
    quant = Quantizer(mode=QuantMode.INFERENCE).to(device)

    from PIL import Image
    import torchvision.transforms as T
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    detector = (
        fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
    )
    for p in detector.parameters():
        p.requires_grad = False

    fpn_key_map = {"P2": "0", "P3": "1", "P4": "2", "P5": "3"}
    fpn_key = fpn_key_map.get(level, "2")

    coco_dir = Path(args.coco_dir)
    val_dir = coco_dir / "val2017"
    ann_file = coco_dir / "annotations/instances_val2017.json"
    if not val_dir.exists():
        print(f"  COCO val2017 not found at {val_dir}")
        return None

    from pycocotools.coco import COCO

    coco_gt = COCO(str(ann_file))
    all_ids = coco_gt.getImgIds()
    random.seed(42)
    random.shuffle(all_ids)
    train_ids, test_ids = all_ids[:200], all_ids[200:400]

    to_tensor = T.ToTensor()

    print(
        f"\n  [probe] Training latent probe on {level} latents ({len(train_ids)} train, {len(test_ids)} test)..."
    )
    probe = LatentProbe(latent_dim=ld).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    probe.train()

    for epoch in range(args.probe_epochs):
        total_loss = 0.0
        for img_id in tqdm(train_ids, desc=f"Epoch {epoch + 1}"):
            fpath = val_dir / f"{img_id:012d}.jpg"
            if not fpath.exists():
                continue
            img = to_tensor(Image.open(fpath).convert("RGB")).to(device)
            img_list, _ = detector.transform([img])
            x = img_list.tensors
            fpn_feats = detector.backbone(x)
            with torch.no_grad():
                latent = comp(fpn_feats[fpn_key])
                qz = quant(latent)
            has_obj = 1.0 if coco_gt.getAnnIds(imgIds=img_id) else 0.0
            pred = probe(qz)
            loss = F.binary_cross_entropy(pred, torch.tensor([has_obj], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"    Loss: {total_loss / max(len(train_ids), 1):.4f}")

    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img_id in test_ids:
            fpath = val_dir / f"{img_id:012d}.jpg"
            if not fpath.exists():
                continue
            img = to_tensor(Image.open(fpath).convert("RGB")).to(device)
            img_list, _ = detector.transform([img])
            x = img_list.tensors
            fpn_feats = detector.backbone(x)
            latent = comp(fpn_feats[fpn_key])
            qz = quant(latent)
            pred = (probe(qz) > 0.5).float().item()
            label = 1.0 if coco_gt.getAnnIds(imgIds=img_id) else 0.0
            correct += int(pred == label)
            total += 1
    acc = correct / max(total, 1)
    print(f"  [probe] Latent probe accuracy: {acc:.2%} ({correct}/{total})")
    return {"probe_accuracy": acc}


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="FPN Compressor Validation Suite")
    parser.add_argument("--mode", type=str, default="all", choices=["recon", "map", "probe", "all"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--fpn-level", type=str, default=None, help="FPN level (default: from checkpoint or P4)"
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="datasets/fpn_features",
        help="Directory with pre-extracted FPN features (for recon mode)",
    )
    parser.add_argument(
        "--coco-dir",
        type=str,
        default="datasets/coco",
        help="COCO dataset root (for map/probe modes)",
    )
    parser.add_argument(
        "--num-images", type=int, default=500, help="Number of COCO val images to evaluate"
    )
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    results = {}
    modes = ["recon", "map", "probe"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'=' * 60}")
        if mode == "recon":
            results["recon"] = run_recon(args)
        elif mode == "map":
            results["map"] = run_map(args)
        elif mode == "probe":
            results["probe"] = run_probe(args)

    out_path = Path(args.ckpt).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
