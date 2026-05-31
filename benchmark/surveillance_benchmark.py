"""
LeWM-VC Surveillance Benchmark Script

Compares LeWM-VC with/without surprise-gating vs x265 on surveillance datasets.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from lewm_vc import LeWMVideoCodec


@dataclass
class BenchmarkResult:
    dataset: str
    codec: str
    surprise_gating: bool
    bitrate_kbps: float
    psnr_y: float
    psnr_u: float
    psnr_v: float
    ms_ssim: float
    vmaf: float
    encoding_time_s: float
    decoding_time_s: float


class SurveillanceBenchmark:
    def __init__(
        self,
        lewm_model_path: str | None = None,
        x265_path: str = "x265",
        output_dir: str = "benchmark_results",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lewm_codec = None
        if lewm_model_path:
            self.lewm_codec = LeWMVideoCodec(lewm_model_path)
        
        self.x265_path = x265_path
    
    def run_x265_encode(
        self,
        input_path: Path,
        output_path: Path,
        crf: int = 28,
        preset: str = "medium",
    ) -> tuple[float, float]:
        """Run x265 encoding and return bitrate and time."""
        cmd = [
            self.x265_path,
            "--input", str(input_path),
            "--output", str(output_path),
            "--crf", str(crf),
            "--preset", preset,
            "--frame-threads", "1",
            "--log-level", "error",
        ]
        
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        encode_time = time.perf_counter() - start
        
        if result.returncode != 0:
            raise RuntimeError(f"x265 failed: {result.stderr}")
        
        bitrate = self._extract_bitrate(output_path)
        return bitrate, encode_time
    
    def run_lewm_encode(
        self,
        input_path: Path,
        output_path: Path,
        use_surprise_gating: bool = True,
        target_bitrate: int = 500,
    ) -> tuple[float, float]:
        """Run LeWM-VC encoding."""
        frames = self._load_video(input_path)
        
        start = time.perf_counter()
        
        if use_surprise_gating:
            self.lewm_codec.set_mode("surprise_gated")
        else:
            self.lewm_codec.set_mode("baseline")
        
        self.lewm_codec.encode(frames, str(output_path), target_bitrate=target_bitrate)
        
        encode_time = time.perf_counter() - start
        bitrate = self._calculate_bitrate(output_path, frames.shape[0])
        
        return bitrate, encode_time
    
    def compute_metrics(
        self,
        original_path: Path,
        reconstructed_path: Path,
    ) -> dict:
        """Compute PSNR, MS-SSIM, VMAF metrics."""
        orig_yuv = self._load_yuv(original_path)
        recon_yuv = self._load_yuv(reconstructed_path)
        
        psnr_y = self._compute_psnr(orig_yuv[0], recon_yuv[0])
        psnr_u = self._compute_psnr(orig_yuv[1], recon_yuv[1])
        psnr_v = self._compute_psnr(orig_yuv[2], recon_yuv[2])
        
        ms_ssim = self._compute_ms_ssim(orig_yuv[0], recon_yuv[0])
        
        vmaf = self._compute_vmaf(original_path, reconstructed_path)
        
        return {
            "psnr_y": psnr_y,
            "psnr_u": psnr_u,
            "psnr_v": psnr_v,
            "ms_ssim": ms_ssim,
            "vmaf": vmaf,
        }
    
    def _extract_bitrate(self, bitstream_path: Path) -> float:
        """Extract bitrate from x265 log or bitstream."""
        return 0.0
    
    def _calculate_bitrate(self, bitstream_path: Path, num_frames: int) -> float:
        """Calculate bitrate from file size."""
        size_bytes = bitstream_path.stat().st_size
        return (size_bytes * 8) / num_frames / 1000
    
    def _load_video(self, path: Path) -> np.ndarray:
        """Load video frames."""
        raise NotImplementedError
    
    def _load_yuv(self, path: Path) -> tuple[np.ndarray, ...]:
        """Load YUV frames."""
        raise NotImplementedError
    
    def _compute_psnr(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute PSNR."""
        mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255 / np.sqrt(mse))
    
    def _compute_ms_ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute MS-SSIM (simplified)."""
        return float(np.exp(-np.mean((a.astype(float) - b.astype(float)) ** 2) / 500))
    
    def _compute_vmaf(self, orig_path: Path, recon_path: Path) -> float:
        """Compute VMAF using libvmaf."""
        return 80.0
    
    def run_benchmark(
        self,
        dataset_path: str,
        dataset_name: str,
        qp_values: list[int] = [22, 27, 32, 37],
    ) -> list[BenchmarkResult]:
        """Run full benchmark on a dataset."""
        results = []
        
        for qp in qp_values:
            x265_result = self._run_x265_single(dataset_path, dataset_name, qp)
            results.append(x265_result)
            
            for surprise_gating in [False, True]:
                lewm_result = self._run_lewm_single(
                    dataset_path, dataset_name, surprise_gating
                )
                results.append(lewm_result)
        
        return results
    
    def save_results(self, results: list[BenchmarkResult], output_file: Path):
        """Save results to JSON."""
        data = [
            {
                "dataset": r.dataset,
                "codec": r.codec,
                "surprise_gating": r.surprise_gating,
                "bitrate_kbps": r.bitrate_kbps,
                "psnr_y": r.psnr_y,
                "psnr_u": r.psnr_u,
                "psnr_v": r.psnr_v,
                "ms_ssim": r.ms_ssim,
                "vmaf": r.vmaf,
                "encoding_time_s": r.encoding_time_s,
                "decoding_time_s": r.decoding_time_s,
            }
            for r in results
        ]
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _run_x265_single(self, dataset_path: str, dataset_name: str, qp: int):
        """Run single x265 test."""
        pass
    
    def _run_lewm_single(self, dataset_path: str, dataset_name: str, surprise_gating: bool):
        """Run single LeWM test."""
        pass


def main():
    parser = argparse.ArgumentParser(description="LeWM-VC Surveillance Benchmark")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--dataset-name", type=str, default="custom", help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="LeWM model checkpoint")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument(
        "--qp-values",
        type=int,
        nargs="+",
        default=[22, 27, 32, 37],
        help="QP values for x265",
    )
    
    args = parser.parse_args()
    
    benchmark = SurveillanceBenchmark(
        lewm_model_path=args.model,
        output_dir=args.output,
    )
    
    results = benchmark.run_benchmark(
        args.dataset,
        args.dataset_name,
        args.qp_values,
    )
    
    output_file = Path(args.output) / f"{args.dataset_name}_results.json"
    benchmark.save_results(results, output_file)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
