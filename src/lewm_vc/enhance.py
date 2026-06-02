import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def build_gaussian_cdf(mu, sigma, step_size, num_levels=256):
    B, C, H, W = mu.shape
    device = mu.device
    half = num_levels // 2
    offsets = (
        torch.arange(0, num_levels + 1, dtype=torch.float32, device=device) - half - 0.5
    ) * step_size
    z = (offsets - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    cdf_float = torch.special.ndtr(z)
    cdf_min = cdf_float[..., 0:1]
    cdf_max = cdf_float[..., -1:]
    cdf_float = (cdf_float - cdf_min) / (cdf_max - cdf_min + 1e-10)
    cdf_float[..., 0] = 0.0
    cdf_float[..., -1] = 1.0
    cdf_int = (cdf_float * 65535 + 0.5).to(torch.int32)
    cdf_int = torch.where(cdf_int > 32767, cdf_int - 65536, cdf_int).to(torch.int16)
    cdf_int[..., -1] = -1
    return cdf_int


def load_models(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    mc = config.get("model", {})

    from lewm_vc.feature_compress import (
        FeatureCompressor,
        FeatureDecompressor,
        ResNetFeatureExtractor,
    )
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer, QuantMode

    backbone_name = mc.get("backbone", "resnet18")
    latent_dim = mc.get("latent_dim", 8)

    backbone = ResNetFeatureExtractor(backbone_name=backbone_name).to(device).eval()
    feat_c = backbone.feature_channels

    compressor = FeatureCompressor(in_channels=feat_c, latent_dim=latent_dim).to(device).eval()
    decompressor = FeatureDecompressor(latent_dim=latent_dim, out_channels=feat_c).to(device).eval()
    entropy_model = (
        HyperpriorEntropy(
            latent_dim=latent_dim,
            hyper_channels=mc.get("entropy", {}).get("hyper_channels", 32),
        )
        .to(device)
        .eval()
    )

    compressor.load_state_dict(ckpt["models"]["compressor"])
    decompressor.load_state_dict(ckpt["models"]["decompressor"])
    entropy_model.load_state_dict(ckpt["models"]["entropy_model"])

    quantizer = Quantizer()
    quantizer.set_mode(QuantMode.INFERENCE)
    quantizer.to(device)

    return {
        "backbone": backbone,
        "compressor": compressor,
        "decompressor": decompressor,
        "entropy_model": entropy_model,
        "quantizer": quantizer,
        "info": {
            "backbone": backbone_name,
            "latent_dim": latent_dim,
            "feature_channels": feat_c,
            "step_size": quantizer.step_size.item(),
        },
    }


class EnhancementRecorder:
    """H.265 enhancement layer recorder using hardware VideoToolbox encoder."""

    def __init__(
        self,
        output_dir: str,
        camera_id: str = "cam1",
        fps: int = 30,
        segment_duration: int = 300,
        quality: int = 70,
        resolution: tuple = (256, 256),
    ):
        self.output_dir = Path(output_dir) / camera_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.fps = fps
        self.segment_duration = segment_duration
        self.quality = quality
        self.resolution = resolution
        self.metadata_path = self.output_dir / "metadata.json"
        self._load_metadata()
        self._current_segment = None
        self._ffmpeg_proc = None
        self._segment_start = None
        self._frame_count = 0

    def _load_metadata(self):
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text())
        else:
            self.metadata = {"segments": []}

    def _save_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2))

    def _segment_filename(self, start_time: float) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))
        return f"{ts}_{self.camera_id}.h265"

    def _start_new_segment(self, frame: np.ndarray):
        self._close_segment()
        self._segment_start = time.time()
        fname = self._segment_filename(self._segment_start)
        fpath = self.output_dir / fname
        w, h = self.resolution

        self._ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{w}x{h}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-c:v",
                "hevc_videotoolbox",
                "-q:v",
                str(self.quality),
                "-tag:v",
                "hvc1",
                str(fpath),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._frame_count = 0
        return fpath

    def _close_segment(self):
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.stdin.close()
            self._ffmpeg_proc.wait()
            duration = time.time() - self._segment_start
            self.metadata["segments"].append(
                {
                    "camera_id": self.camera_id,
                    "start_time": self._segment_start,
                    "end_time": time.time(),
                    "duration": round(duration, 1),
                    "frames": self._frame_count,
                    "fps": self.fps,
                    "resolution": list(self.resolution),
                }
            )
            self._save_metadata()
            self._ffmpeg_proc = None

    def write_frame(self, frame: np.ndarray):
        if self._ffmpeg_proc is None:
            self._start_new_segment(frame)
        elif self._frame_count >= self.fps * self.segment_duration:
            self._start_new_segment(frame)
        self._ffmpeg_proc.stdin.write(frame.tobytes())
        self._frame_count += 1

    def close(self):
        self._close_segment()

    def list_segments(self):
        return self.metadata.get("segments", [])

    def extract_segment(self, segment_idx: int, output_path: str):
        segments = self.metadata.get("segments", [])
        if segment_idx < 0 or segment_idx >= len(segments):
            print(f"Invalid segment index {segment_idx}. Available: 0-{len(segments) - 1}")
            return
        seg = segments[segment_idx]
        fname = (
            self._segment_filename(seg["start_time"]) if "filename" not in seg else seg["filename"]
        )
        src = self.output_dir / fname
        if not src.exists():
            print(f"Segment file not found: {src}")
            return
        dst = Path(output_path)
        dst.write_bytes(src.read_bytes())
        print(f"Extracted segment {segment_idx} ({round(seg['duration'], 1)}s) to {output_path}")


def cmd_enhance_record(args):
    import torchac

    frames_dir = Path(args.input)
    frames = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        print(f"No frames found in {args.input}")
        return

    if args.compress and not args.model:
        print("--model is required when --compress is set")
        return

    models = None
    if args.compress:
        models = load_models(args.model, args.device)

    resolution = (args.image_size, args.image_size)
    recorder = EnhancementRecorder(
        output_dir=args.output,
        camera_id=args.camera_id,
        fps=args.fps,
        segment_duration=args.segment_duration,
        quality=args.encode_quality,
        resolution=resolution,
    )

    n_pixels = args.image_size * args.image_size
    total_base_bytes = 0
    print(f"Recording {len(frames)} frames...")
    t0 = time.time()

    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize(resolution)
        frame_np = np.array(img)
        recorder.write_frame(frame_np)

        if args.compress:
            device = args.device
            tensor = (
                torch.from_numpy(frame_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            )
            with torch.no_grad():
                feats = models["backbone"](tensor)
                latent = models["compressor"](feats)
                qz = models["quantizer"](latent)
                step_size = models["info"]["step_size"]
                indices = torch.round(qz / step_size).clamp(-128, 127).to(torch.int16) + 128
                indices = indices.to(torch.int16).cpu()
                _, params = models["entropy_model"](qz)
                mu = params["mu"].detach().cpu()
                sigma = params["sigma"].detach().cpu()
                cdf = build_gaussian_cdf(mu, sigma, step_size)
                encoded = torchac.encode_int16_normalized_cdf(cdf, indices)
                total_base_bytes += len(encoded)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                avg_bpp = total_base_bytes * 8 / n_pixels / (i + 1)
                print(
                    f"  [{i + 1}/{len(frames)}]  base_bpp={avg_bpp:.4f}  enhance_h265  [{elapsed:.0f}s]"
                )

    recorder.close()
    elapsed = time.time() - t0
    avg_bpp = total_base_bytes * 8 / n_pixels / len(frames) if args.compress else 0
    segments = recorder.list_segments()
    total_enhance_bytes = sum(
        (recorder.output_dir / recorder._segment_filename(s["start_time"])).stat().st_size
        for s in segments
    )
    print(f"\nDone. {len(frames)} frames in {elapsed:.1f}s")
    print(f"  Base layer:    {total_base_bytes} bytes, {avg_bpp:.4f} avg BPP")
    print(f"  Enhancement:   {total_enhance_bytes} bytes, {len(segments)} segments")
    print(f"  Stored at:     {recorder.output_dir}")
    if segments:
        print(f"  Segments:")
        for i, s in enumerate(segments):
            print(f"    [{i}] {round(s['duration'], 1)}s, {s['frames']} frames")


def cmd_enhance_list(args):
    from lewm_vc.enhance import EnhancementRecorder

    recorder = EnhancementRecorder(output_dir=args.store, camera_id=args.camera_id)
    segments = recorder.list_segments()
    if not segments:
        print(
            f"No enhancement segments found for camera '{args.camera_id}' at {recorder.output_dir}"
        )
        return
    print(f"Enhancement segments for camera '{args.camera_id}':")
    for i, s in enumerate(segments):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(s["start_time"]))
        print(
            f"  [{i}] {ts}  {round(s['duration'], 1)}s  {s['frames']} frames  {list(s['resolution'])}"
        )


def cmd_enhance_extract(args):
    from lewm_vc.enhance import EnhancementRecorder

    recorder = EnhancementRecorder(output_dir=args.store, camera_id=args.camera_id)
    recorder.extract_segment(args.segment, args.output)
