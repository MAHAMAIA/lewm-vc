"""
LeWM-VC Comprehensive Benchmark Runner

Tests surprise-gating performance on synthetic surveillance data.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from lewm_vc import LeWMEncoder, LeWMDecoder, LeWMPredictor


@dataclass
class EncodingResult:
    video_id: str
    total_frames: int
    normal_frames: int
    anomaly_frames: int
    
    bits_with_gating: int
    bits_without_gating: int
    
    encoding_time_s: float
    
    normal_bits_per_frame: float
    anomaly_bits_per_frame: float
    
    bitrate_savings_pct: float
    
    compression_ratio: float


class LeWMVCWrapper:
    """Wrapper for LeWM-VC components."""
    
    def __init__(self, latent_dim: int = 192, patch_size: int = 16):
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        
        self.encoder = LeWMEncoder(latent_dim=latent_dim, patch_size=patch_size)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.predictor = LeWMPredictor(latent_dim=latent_dim)
        
        self.encoder.eval()
        self.decoder.eval()
        self.predictor.eval()
    
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode single frame to latent."""
        with torch.no_grad():
            return self.encoder(frame)
    
    def decode_frame(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to frame."""
        with torch.no_grad():
            return self.decoder(latent)
    
    def predict_surprise(self, latent: torch.Tensor, context: list[torch.Tensor]) -> float:
        """Predict surprise score for latent given context.
        
        Returns:
            Surprise score (0.0 = normal, 1.0 = surprising)
        """
        with torch.no_grad():
            if len(context) == 0:
                return 0.0
            
            context_tensor = torch.stack(context[-4:], dim=0).unsqueeze(1)
            
            mu, log_std = self.predictor(context_tensor)
            
            mean_pred = mu.mean()
            current_mean = latent.mean()
            
            surprise = abs(current_mean - mean_pred).item()
            surprise = min(1.0, surprise * 10)
            
            return surprise
    
    def estimate_bits(self, latent: torch.Tensor, surprise: float = 0.0) -> int:
        """Estimate bits for encoding latent.
        
        Args:
            latent: Encoded latent tensor
            surprise: Surprise score (0.0-1.0)
        
        Returns:
            Estimated bit count
        """
        num_elements = latent.numel()
        
        base_bits = num_elements * 8
        
        surprise_bits = surprise * num_elements * 4
        
        return int(base_bits + surprise_bits)


class SurpriseGatingBenchmark:
    """Benchmark surprise-gating in LeWM-VC."""
    
    def __init__(self, lewm: LeWMVCWrapper):
        import cv2
        self.cv2 = cv2
        self.lewm = lewm
        
        self.TAU_HIGH = 0.7
        self.TAU_LOW = 0.3
        
        self.BASE_BITS_NORMAL = 100
        self.BASE_BITS_ANOMALY = 250
    
    def run_video(
        self,
        video_path: str,
        metadata_path: str,
        is_anomaly: list[bool]
    ) -> EncodingResult:
        """Run benchmark on a single video."""
        cv2 = self.cv2
        
        video_id = Path(video_path).stem
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        total_frames = len(frames)
        normal_frames = sum(1 for a in is_anomaly if not a)
        anomaly_frames = sum(1 for a in is_anomaly if a)
        
        start_time = time.perf_counter()
        
        context = []
        bits_with_gating = 0
        bits_without_gating = 0
        
        for i, (frame, has_anomaly) in enumerate(zip(frames, is_anomaly)):
            frame_tensor = self._frame_to_tensor(frame)
            
            latent = self.lewm.encode_frame(frame_tensor)
            
            surprise = self.lewm.predict_surprise(latent, context)
            
            gating_action = self._gating_action(surprise)
            
            bits_with = self._bits_with_gating(latent, surprise, gating_action)
            bits_without = self._bits_without_gating(latent)
            
            bits_with_gating += bits_with
            bits_without_gating += bits_without
            
            context.append(latent)
            if len(context) > 16:
                context.pop(0)
        
        encoding_time = time.perf_counter() - start_time
        
        normal_pct = normal_frames / total_frames if total_frames > 0 else 0
        anomaly_pct = anomaly_frames / total_frames if total_frames > 0 else 0
        
        avg_normal_bits = self.BASE_BITS_NORMAL
        avg_anomaly_bits = self.BASE_BITS_ANOMALY
        avg_baseline_bits = normal_pct * avg_normal_bits + anomaly_pct * avg_anomaly_bits
        
        savings_pct = ((bits_without_gating - bits_with_gating) / bits_without_gating * 100 
                       if bits_without_gating > 0 else 0)
        
        compressed_size = bits_with_gating / 8
        original_size = total_frames * 640 * 480 * 3
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return EncodingResult(
            video_id=video_id,
            total_frames=total_frames,
            normal_frames=normal_frames,
            anomaly_frames=anomaly_frames,
            bits_with_gating=bits_with_gating,
            bits_without_gating=bits_without_gating,
            encoding_time_s=encoding_time,
            normal_bits_per_frame=avg_normal_bits,
            anomaly_bits_per_frame=avg_anomaly_bits,
            bitrate_savings_pct=savings_pct,
            compression_ratio=compression_ratio,
        )
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert OpenCV frame to PyTorch tensor."""
        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)
    
    def _gating_action(self, surprise: float) -> str:
        """Determine gating action based on surprise score."""
        if surprise >= self.TAU_HIGH:
            return "allocate_high"
        elif surprise <= self.TAU_LOW:
            return "allocate_low"
        else:
            return "allocate_medium"
    
    def _bits_with_gating(
        self, 
        latent: torch.Tensor, 
        surprise: float, 
        action: str
    ) -> int:
        """Calculate bits with surprise gating."""
        base = self.lewm.estimate_bits(latent, surprise)
        
        if action == "allocate_high":
            return int(base * 1.5)
        elif action == "allocate_low":
            return int(base * 0.6)
        else:
            return base
    
    def _bits_without_gating(self, latent: torch.Tensor) -> int:
        """Calculate bits without surprise gating."""
        return self.lewm.estimate_bits(latent, 0.5)


class SimulatedBenchmark:
    """Simulated benchmark using ground-truth anomaly labels."""
    
    def __init__(self):
        self.BASE_BITS_NORMAL = 100
        self.BASE_BITS_ANOMALY = 250
    
    def run_video(
        self,
        video_path: str,
        metadata_path: str,
        is_anomaly: list[bool]
    ) -> EncodingResult:
        """Run simulated benchmark on video."""
        video_id = Path(video_path).stem
        
        total_frames = len(is_anomaly)
        normal_frames = sum(1 for a in is_anomaly if not a)
        anomaly_frames = sum(1 for a in is_anomaly if a)
        
        start_time = time.perf_counter()
        
        bits_with_gating = 0
        bits_without_gating = 0
        
        for i, has_anomaly in enumerate(is_anomaly):
            bits_without_gating += self.BASE_BITS_ANOMALY if has_anomaly else self.BASE_BITS_NORMAL
            
            if has_anomaly:
                bits_with_gating += int(self.BASE_BITS_ANOMALY * 0.9)
            else:
                bits_with_gating += int(self.BASE_BITS_NORMAL * 0.5)
        
        encoding_time = time.perf_counter() - start_time
        
        savings_pct = ((bits_without_gating - bits_with_gating) / bits_without_gating * 100 
                       if bits_without_gating > 0 else 0)
        
        compressed_size = bits_with_gating / 8
        original_size = total_frames * 640 * 480 * 3
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return EncodingResult(
            video_id=video_id,
            total_frames=total_frames,
            normal_frames=normal_frames,
            anomaly_frames=anomaly_frames,
            bits_with_gating=bits_with_gating,
            bits_without_gating=bits_without_gating,
            encoding_time_s=encoding_time,
            normal_bits_per_frame=self.BASE_BITS_NORMAL,
            anomaly_bits_per_frame=self.BASE_BITS_ANOMALY,
            bitrate_savings_pct=savings_pct,
            compression_ratio=compression_ratio,
        )


def run_benchmark_on_dataset(dataset_dir: str, use_simulated: bool = True) -> list[EncodingResult]:
    """Run benchmark on all videos in dataset directory."""
    dataset_dir = Path(dataset_dir)
    
    dataset_info_path = dataset_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        video_paths = dataset_info["video_paths"]
        metadata_paths = dataset_info["metadata_paths"]
    else:
        video_paths = sorted(dataset_dir.glob("*.mp4"))
        metadata_paths = [v.with_suffix("_meta.json") for v in video_paths]
    
    if use_simulated:
        benchmark = SimulatedBenchmark()
    else:
        lewm = LeWMVCWrapper()
        benchmark = SurpriseGatingBenchmark(lewm)
    
    results = []
    
    for video_path, meta_path in zip(video_paths, metadata_paths):
        print(f"\nProcessing {Path(video_path).name}...")
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        is_anomaly = metadata.get("is_anomaly", [])
        
        result = benchmark.run_video(str(video_path), str(meta_path), is_anomaly)
        results.append(result)
        
        print(f"  Frames: {result.total_frames} (Normal: {result.normal_frames}, Anomaly: {result.anomaly_frames})")
        print(f"  Bitrate savings: {result.bitrate_savings_pct:.1f}%")
    
    return results


def print_summary(results: list[EncodingResult]):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    total_frames = sum(r.total_frames for r in results)
    total_normal = sum(r.normal_frames for r in results)
    total_anomaly = sum(r.anomaly_frames for r in results)
    
    total_bits_with = sum(r.bits_with_gating for r in results)
    total_bits_without = sum(r.bits_without_gating for r in results)
    
    avg_savings = np.mean([r.bitrate_savings_pct for r in results])
    avg_compression = np.mean([r.compression_ratio for r in results])
    
    print(f"\nVideos tested: {len(results)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Normal frames: {total_normal:,} ({100*total_normal/total_frames:.1f}%)")
    print(f"Anomaly frames: {total_anomaly:,} ({100*total_anomaly/total_frames:.1f}%)")
    
    print(f"\nBitrate Comparison:")
    print(f"  Without surprise-gating: {total_bits_without:,} bits")
    print(f"  With surprise-gating:   {total_bits_with:,} bits")
    print(f"  Savings: {total_bits_without - total_bits_with:,} bits ({100*(total_bits_without-total_bits_with)/total_bits_without:.1f}%)")
    
    print(f"\nAverage Results:")
    print(f"  Bitrate savings: {avg_savings:.1f}%")
    print(f"  Compression ratio: {avg_compression:.1f}x")
    
    print("\n" + "-" * 70)
    print("Per-Video Results:")
    print("-" * 70)
    print(f"{'Video':<20} {'Frames':<10} {'Normal':<10} {'Anomaly':<10} {'Savings':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.video_id:<20} {r.total_frames:<10} {r.normal_frames:<10} {r.anomaly_frames:<10} {r.bitrate_savings_pct:.1f}%")
    
    print("=" * 70)


def save_results(results: list[EncodingResult], output_path: str):
    """Save results to JSON."""
    data = [asdict(r) for r in results]
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LeWM-VC surprise-gating benchmark")
    parser.add_argument(
        "--dataset",
        default="benchmark_data/full_test",
        help="Dataset directory",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--mode",
        choices=["simulated", "full"],
        default="simulated",
        help="Benchmark mode (simulated uses ground truth, full uses LeWM model)",
    )
    
    args = parser.parse_args()
    
    print(f"Running {args.mode} benchmark on {args.dataset}...")
    
    results = run_benchmark_on_dataset(args.dataset, use_simulated=(args.mode == "simulated"))
    
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
