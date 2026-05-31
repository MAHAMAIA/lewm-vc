#!/usr/bin/env python3
"""
Dataset validation script.

Scans data roots from config, computes per-clip stats (motion, frame count,
resolution), detects corrupt frames, and generates an HTML report.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --config configs/train_config.yaml
    python scripts/validate_dataset.py --output validation_output --max-clips 50
"""

import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _discover_clips(roots: list[str]) -> list[Path]:
    clips = []
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            print(f"  [WARN] root does not exist: {root}")
            continue
        for entry in sorted(root_p.iterdir()):
            if entry.is_dir():
                frames = sorted(entry.glob("*.png"))
                if frames:
                    clips.append(entry)
    return clips


def _infer_resolution(frames: list[Path]) -> tuple[int, int] | None:
    try:
        with Image.open(frames[0]) as im:
            return im.size
    except Exception:
        return None


def _compute_motion_score(frames: list[Path], max_samples: int = 20) -> float:
    """Mean absolute pixel difference between consecutive frames (0-255 scale)."""
    step = max(1, len(frames) // max_samples)
    diffs = []
    prev = None
    for f in frames[::step]:
        try:
            arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
            if prev is not None:
                diffs.append(np.abs(arr - prev).mean())
            prev = arr
        except Exception:
            continue
    return float(np.mean(diffs)) if diffs else 0.0


def _check_corrupt(frames: list[Path], sample: int = 5) -> list[str]:
    """Check for corrupt or all-black frames."""
    issues = []
    for f in frames[:: max(1, len(frames) // sample)]:
        try:
            arr = np.array(Image.open(f))
            if arr.max() == 0:
                issues.append(f"{f.name}: all-black frame")
            if arr.ndim == 2:
                issues.append(f"{f.name}: grayscale (expected RGB)")
            elif arr.shape[2] not in (3, 4):
                issues.append(f"{f.name}: unexpected channels={arr.shape[2]}")
        except Exception as e:
            issues.append(f"{f.name}: corrupt ({e})")
    return issues


def _frame_to_b64(frame: Image.Image, size: tuple[int, int] = (320, 180)) -> str:
    frame.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate(config_path: str, output_dir: str, max_clips: int | None):
    config = _load_config(config_path)
    data_cfg = config.get("data", {})
    roots = data_cfg.get("roots", [])
    image_size = data_cfg.get("image_size", 256)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Config: {config_path}")
    print(f"Roots: {roots}")
    print(f"Output: {out.resolve()}\n")

    clips = _discover_clips(roots)
    print(f"Total clip directories found: {len(clips)}")

    if not clips:
        print("ERROR: no clips found — check data roots")
        sys.exit(1)

    if max_clips:
        clips = clips[:max_clips]
        print(f"Sampling first {max_clips} clips for detailed analysis")

    # Per-clip analysis
    clip_stats = []
    all_motions = []
    total_frames = 0
    bad_clips = []

    for i, clip_dir in enumerate(clips):
        frames = sorted(clip_dir.glob("*.png"))
        n_frames = len(frames)
        total_frames += n_frames
        resolution = _infer_resolution(frames)
        motion = _compute_motion_score(frames)
        all_motions.append(motion)
        issues = _check_corrupt(frames)

        clip_stats.append(
            {
                "name": clip_dir.name,
                "parent": clip_dir.parent.name,
                "frames": n_frames,
                "resolution": f"{resolution[0]}x{resolution[1]}" if resolution else "unknown",
                "motion": motion,
                "issues": issues,
            }
        )
        if issues:
            bad_clips.append(clip_dir.name)

        if (i + 1) % 20 == 0:
            print(f"  scanned {i + 1}/{len(clips)} clips...")

    # Summary stats
    motions = np.array(all_motions)
    print(f"\nTotal frames: {total_frames}")
    print(
        f"Motion score: mean={motions.mean():.2f}  std={motions.std():.2f}  "
        f"min={motions.min():.2f}  max={motions.max():.2f}"
    )
    print(f"Clips with issues: {len(bad_clips)}")

    # Test FrameDataset forward pass
    if torch is not None:
        print("\n--- Testing FrameDataset forward pass ---")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from lewm_vc.data import FrameDataset, collate_sequences

            ds = FrameDataset.from_roots(
                roots,
                sequence_length=4,
                image_size=image_size,
                augment=False,
                split="train",
            )
            loader = torch.utils.data.DataLoader(
                ds, batch_size=2, shuffle=False, collate_fn=collate_sequences
            )
            batch = next(iter(loader))
            frames_t = batch["frames"]
            assert frames_t.ndim == 5, f"expected [B,T,3,H,W], got {frames_t.shape}"
            assert frames_t.shape[1] == 4, f"expected T=4, got {frames_t.shape[1]}"
            print(f"  FrameDataset OK: batch shape {tuple(frames_t.shape)}")
            print(f"  Value range: [{frames_t.min():.3f}, {frames_t.max():.3f}]")
        except Exception as e:
            print(f"  FrameDataset ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Generate HTML report
    print("\n--- Generating HTML report ---")
    _generate_report(clip_stats, motions, bad_clips, out, image_size)
    print(f"Report: {out / 'report.html'}")


def _generate_report(
    clip_stats: list[dict],
    motions: np.ndarray,
    bad_clips: list[str],
    out: Path,
    image_size: int,
):
    # Clip table rows
    table_rows = ""
    for s in clip_stats[:200]:
        issue_badge = ""
        if s["issues"]:
            issue_badge = '<span class="badge bad">issues</span>'
        motion_bar = min(s["motion"] / max(motions.max(), 1.0), 1.0)
        table_rows += (
            f"<tr>"
            f"<td>{s['name']}</td>"
            f"<td>{s['parent']}</td>"
            f"<td>{s['frames']}</td>"
            f"<td>{s['resolution']}</td>"
            f"<td>"
            f"  <div style='width:80px;height:8px;background:#eee;border-radius:4px;'>"
            f"    <div style='width:{motion_bar * 80:.0f}px;height:8px;"
            f"background:#4f8;border-radius:4px;'></div>"
            f"  </div>"
            f"  {s['motion']:.1f}"
            f"</td>"
            f"<td>{issue_badge}</td>"
            f"</tr>\n"
        )

    # Sample clips grid (first frame of first 12 clips)
    sample_cells = ""
    for s in clip_stats[:12]:
        clip_dir = Path(s["parent"]) / s["name"]
        frames = sorted(clip_dir.glob("*.png"))
        if frames:
            try:
                img = Image.open(frames[0])
                b64 = _frame_to_b64(img)
                sample_cells += (
                    f"<div class='sample'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    f"alt='{s['name']}' title='{s['name']}: {s['frames']} frames'>"
                    f"<div class='label'>{s['name']}</div>"
                    f"</div>\n"
                )
            except Exception:
                pass

    # Motion histogram bins
    hist_bins = 10
    counts, edges = np.histogram(motions, bins=hist_bins)
    hist_bar = ""
    max_count = max(counts.max(), 1)
    for i in range(hist_bins):
        pct = counts[i] / max_count * 100
        left = edges[i]
        hist_bar += (
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<span style='width:50px;font-size:11px;'>{left:.1f}</span>"
            f"<div style='width:{pct:.0f}%;max-width:200px;height:14px;"
            f"background:#48f;border-radius:3px;'></div>"
            f"<span style='margin-left:6px;font-size:11px;'>{counts[i]}</span>"
            f"</div>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dataset Validation Report — LeWM-VC</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 20px; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a1a2e; }}
.summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: white; padding: 16px 20px; border-radius: 8px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 140px; }}
.card .num {{ font-size: 28px; font-weight: 700; color: #1a1a2e; }}
.card .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
          font-size: 11px; font-weight: 600; }}
.badge.bad {{ background: #fdd; color: #c00; }}
table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
         overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #1a1a2e; color: white; padding: 10px 12px; text-align: left;
      font-size: 12px; text-transform: uppercase; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
tr:hover {{ background: #f0f4ff; }}
.samples {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px; margin: 16px 0; }}
.sample {{ background: white; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
.sample img {{ width: 100%; height: auto; display: block; }}
.sample .label {{ padding: 6px; font-size: 11px; color: #555;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.warn {{ background: #fff3cd; border: 1px solid #ffc107; padding: 12px 16px;
         border-radius: 8px; margin: 16px 0; font-size: 13px; }}
.hist {{ margin: 16px 0; }}
.footer {{ margin-top: 24px; font-size: 11px; color: #999; text-align: center; }}
</style>
</head>
<body>
<h1>Dataset Validation Report</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

<div class="summary">
  <div class="card"><div class="num">{len(clip_stats)}</div><div class="lbl">Clips</div></div>
  <div class="card"><div class="num">{sum(s["frames"] for s in clip_stats)}</div><div class="lbl">Frames</div></div>
  <div class="card"><div class="num">{motions.mean():.1f}</div><div class="lbl">Mean Motion</div></div>
  <div class="card"><div class="num">{len(bad_clips)}</div><div class="lbl">Clips with Issues</div></div>
</div>

<h2>Sample Clips (first frame)</h2>
<div class="samples">{sample_cells}</div>

<h2>Motion Score Distribution</h2>
<div class="hist">{hist_bar}</div>

<h2>All Clips</h2>
<div style="max-height:500px;overflow-y:auto;">
<table>
<thead><tr>
<th>Clip</th><th>Dataset</th><th>Frames</th><th>Resolution</th><th>Motion</th><th>Issues</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>

{f"<div class='warn'><b>{len(bad_clips)} clip(s) with issues:</b><br>{'<br>'.join(bad_clips[:20])}</div>" if bad_clips else ""}

<div class="footer">LeWM-VC Dataset Validation — MAHAMAIA Systems</div>
</body>
</html>"""

    (out / "report.html").write_text(html)
    print(f"  wrote {len(html)} bytes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for LeWM-VC training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="validation_output")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit clip analysis count")
    args = parser.parse_args()
    validate(args.config, args.output, args.max_clips)


if __name__ == "__main__":
    main()
