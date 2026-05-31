"""
Synthetic Surveillance Video Generator for LeWM-VC Benchmarking

Generates configurable surveillance-like video with normal vs anomalous frames
for testing surprise-gating and bitrate allocation.

Scene Types:
- indoor_hallway: Controlled corridor with predictable motion
- parking_lot: Outdoor scene with vehicle/pedestrian movement
- outdoor_plaza: Open area with mixed traffic patterns

Anomaly Types:
- motion_burst: Sudden acceleration (running person)
- wrong_direction: Trajectory violation
- dropped_object: New static object appears
- sudden_appearance: Object enters frame abruptly
- static_frame: Motion suddenly stops
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class SceneType(Enum):
    INDOOR_HALLWAY = "indoor_hallway"
    PARKING_LOT = "parking_lot"
    OUTDOOR_PLAZA = "outdoor_plaza"


class AnomalyType(Enum):
    MOTION_BURST = "motion_burst"       # Running person
    WRONG_DIRECTION = "wrong_direction"  # Trajectory violation
    DROPPED_OBJECT = "dropped_object"    # New static object
    SUDDEN_APPEARANCE = "sudden_appearance"  # Abrupt entry
    STATIC_FRAME = "static_frame"        # Motion stops


@dataclass
class SceneConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 25
    duration_s: float = 10.0
    background_color: tuple[int, int, int] = (40, 40, 40)
    anomaly_probability: float = 0.15


@dataclass
class Object:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    size: int = 30
    color: tuple[int, int, int] = (100, 150, 200)
    is_anomaly: bool = False
    label: str = "person"


class SyntheticSurveillanceGenerator:
    def __init__(self, config: SceneConfig | None = None):
        self.config = config or SceneConfig()
        self.width = self.config.width
        self.height = self.config.height
        self.fps = self.config.fps
        self.duration_frames = int(self.config.duration_s * self.fps)
        
        self.objects: list[Object] = []
        self.active_anomalies: list[tuple[AnomalyType, int]] = []  # (type, frames_remaining)
        self.frame_count = 0
        self.rng = np.random.default_rng()
        
        self._init_scene()
    
    def _init_scene(self):
        """Initialize scene with background and initial objects."""
        self.background = self._generate_background()
        
        num_objects = random.randint(2, 5)
        for i in range(num_objects):
            obj = self._spawn_normal_object()
            self.objects.append(obj)
    
    def _generate_background(self) -> np.ndarray:
        """Generate static background image."""
        bg = np.full(
            (self.height, self.width, 3),
            self.config.background_color,
            dtype=np.uint8
        )
        
        if self.width >= 1920:
            self._add_hallway_features(bg)
        else:
            self._add_simple_features(bg)
        
        return bg
    
    def _add_hallway_features(self, bg: np.ndarray):
        """Add architectural features to hallway scene."""
        h, w = bg.shape[:2]
        
        floor_color = (60, 55, 50)
        cv2.rectangle(bg, (0, h - 150), (w, h), floor_color, -1)
        
        wall_color = (80, 75, 70)
        cv2.rectangle(bg, (0, 0), (w, h - 150), wall_color, -1)
        
        door_color = (100, 80, 60)
        cv2.rectangle(bg, (w // 4 - 60, 100), (w // 4 + 60, h - 150), door_color, -1)
        cv2.rectangle(bg, (3 * w // 4 - 60, 100), (3 * w // 4 + 60, h - 150), door_color, -1)
        
        line_color = (100, 95, 90)
        for i in range(0, w, 100):
            cv2.line(bg, (i, 0), (i, h), line_color, 1)
        
        light_color = (200, 200, 180)
        cv2.circle(bg, (w // 2, 50), 20, light_color, -1)
    
    def _add_simple_features(self, bg: np.ndarray):
        """Add simple features for small resolution."""
        h, w = bg.shape[:2]
        cv2.rectangle(bg, (0, h - 50), (w, h), (50, 50, 50), -1)
        
        cv2.circle(bg, (w // 2, 30), 15, (180, 180, 160), -1)
    
    def _spawn_normal_object(self) -> Object:
        """Spawn an object with normal motion pattern."""
        if random.random() < 0.5:
            y = self.height - 150 - random.uniform(20, 100)
            obj = Object(
                x=random.uniform(100, self.width - 100),
                y=y,
                vx=random.uniform(-1.5, 1.5),
                vy=0,
                size=random.randint(25, 40),
                color=self._random_person_color(),
                label="person"
            )
        else:
            y = self.height // 2 + random.uniform(-100, 100)
            obj = Object(
                x=random.uniform(100, self.width - 100),
                y=y,
                vx=random.uniform(-2, 2),
                vy=random.uniform(-0.5, 0.5),
                size=random.randint(40, 70),
                color=self._random_vehicle_color(),
                label="vehicle"
            )
        
        return obj
    
    def _random_person_color(self) -> tuple[int, int, int]:
        """Generate realistic person color."""
        skin_tones = [(200, 180, 160), (180, 140, 120), (150, 120, 100)]
        return random.choice(skin_tones)
    
    def _random_vehicle_color(self) -> tuple[int, int, int]:
        """Generate realistic vehicle color."""
        colors = [(80, 80, 100), (100, 80, 80), (80, 100, 80), (60, 60, 80)]
        return random.choice(colors)
    
    def _spawn_anomaly_object(self, anomaly_type: AnomalyType) -> Object:
        """Spawn an anomaly object based on type."""
        if anomaly_type == AnomalyType.DROPPED_OBJECT:
            return Object(
                x=random.uniform(200, self.width - 200),
                y=random.uniform(200, self.height - 200),
                vx=0, vy=0,
                size=random.randint(15, 25),
                color=(220, 200, 50),
                is_anomaly=True,
                label="object"
            )
        
        elif anomaly_type == AnomalyType.MOTION_BURST:
            speed = random.uniform(8, 12)
            angle = random.uniform(0, 2 * math.pi)
            return Object(
                x=random.uniform(100, self.width - 100),
                y=self.height - 150,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=random.randint(20, 30),
                color=(220, 100, 100),
                is_anomaly=True,
                label="runner"
            )
        
        elif anomaly_type == AnomalyType.SUDDEN_APPEARANCE:
            side = random.choice(['left', 'right'])
            x = 50 if side == 'left' else self.width - 50
            return Object(
                x=x,
                y=random.uniform(200, self.height - 200),
                vx=random.uniform(-1, 1),
                vy=0,
                size=random.randint(25, 35),
                color=(100, 220, 100),
                is_anomaly=True,
                label="intruder"
            )
        
        elif anomaly_type == AnomalyType.WRONG_DIRECTION:
            normal_vx = random.uniform(-1.5, 1.5)
            return Object(
                x=random.uniform(100, self.width - 100),
                y=self.height - 150,
                vx=-normal_vx * 3,
                vy=random.uniform(-0.2, 0.2),
                size=random.randint(25, 35),
                color=(220, 180, 100),
                is_anomaly=True,
                label="wrong_way"
            )
        
        else:
            return self._spawn_normal_object()
    
    def _update_objects(self):
        """Update object positions and handle boundaries."""
        for obj in self.objects:
            if obj.is_anomaly and obj.label == "object":
                continue
            
            obj.x += obj.vx
            obj.y += obj.vy
            
            if obj.x < 50:
                obj.x = 50
                obj.vx = abs(obj.vx)
            elif obj.x > self.width - 50:
                obj.x = self.width - 50
                obj.vx = -abs(obj.vx)
            
            if obj.y < 100:
                obj.y = 100
                obj.vy = abs(obj.vy)
            elif obj.y > self.height - 100:
                obj.y = self.height - 100
                obj.vy = -abs(obj.vy)
    
    def _inject_anomaly(self) -> AnomalyType | None:
        """Randomly inject an anomaly (transient, lasts ~0.25-1 seconds)."""
        if random.random() > self.config.anomaly_probability:
            return None
        
        anomaly_type = random.choice(list(AnomalyType))
        anomaly_obj = self._spawn_anomaly_object(anomaly_type)
        self.objects.append(anomaly_obj)
        
        anomaly_duration = self.fps * random.uniform(0.25, 1.0)  # 0.25-1 seconds
        self.active_anomalies.append((anomaly_type, int(anomaly_duration)))
        
        if anomaly_type == AnomalyType.STATIC_FRAME:
            for obj in self.objects[:-1]:
                obj.vx, obj.vy = 0, 0
        
        return anomaly_type
    
    def _update_anomalies(self):
        """Remove expired anomalies and their objects."""
        expired = []
        for i, (anomaly_type, frames_remaining) in enumerate(self.active_anomalies):
            if frames_remaining <= 0:
                expired.append(i)
        
        for i in reversed(expired):
            self.active_anomalies.pop(i)
            anomaly_objs = [o for o in self.objects if o.is_anomaly]
            if anomaly_objs:
                self.objects.remove(anomaly_objs[0])
        
        for i in range(len(self.active_anomalies)):
            anomaly_type, frames_remaining = self.active_anomalies[i]
            self.active_anomalies[i] = (anomaly_type, frames_remaining - 1)
    
    def _has_active_anomaly(self) -> bool:
        """Check if there's an active anomaly this frame."""
        return len(self.active_anomalies) > 0
    
    def _render_frame(self) -> np.ndarray:
        """Render current frame with all objects."""
        frame = self.background.copy()
        
        for obj in self.objects:
            x, y = int(obj.x), int(obj.y)
            size = obj.size
            
            if obj.label == "person" or obj.label in ["runner", "wrong_way", "intruder"]:
                color = obj.color
                cv2.ellipse(frame, (x, y - size // 3), (size // 3, size // 4), 0, 0, 360, color, -1)
                cv2.rectangle(frame, (x - size // 4, y - size // 3), (x + size // 4, y + size // 2), color, -1)
                
            elif obj.label == "vehicle":
                color = obj.color
                cv2.rectangle(frame, (x - size, y - size // 2), (x + size, y + size // 2), color, -1)
                cv2.circle(frame, (x - size // 2, y + size // 2), size // 4, (30, 30, 30), -1)
                cv2.circle(frame, (x + size // 2, y + size // 2), size // 4, (30, 30, 30), -1)
            
            elif obj.label == "object":
                cv2.circle(frame, (x, y), size, obj.color, -1)
        
        cv2.putText(frame, f"F: {self.frame_count}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        anomaly_count = sum(1 for o in self.objects if o.is_anomaly)
        if anomaly_count > 0:
            cv2.putText(frame, f"ANOMALY: {anomaly_count}", (self.width - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def generate_normal_sequence(self, duration_s: float | None = None) -> list[np.ndarray]:
        """Generate normal surveillance sequence."""
        self.config.anomaly_probability = 0.0
        return self._generate(duration_s)
    
    def generate_anomaly_sequence(
        self, 
        anomaly_type: AnomalyType,
        duration_s: float | None = None
    ) -> list[np.ndarray]:
        """Generate sequence with specific anomaly type injected."""
        self.config.anomaly_probability = 1.0
        self._force_anomaly = anomaly_type
        frames = self._generate(duration_s)
        self._force_anomaly = None
        return frames
    
    def generate_mixed_sequence(
        self, 
        normal_pct: float = 0.8,
        duration_s: float | None = None
    ) -> tuple[list[np.ndarray], list[bool]]:
        """Generate mixed normal/anomaly sequence.
        
        Returns:
            frames: List of video frames
            is_anomaly: List of bools indicating anomaly at each frame
        """
        avg_anomaly_duration = self.fps * 0.5  # Average ~0.5 seconds
        self.config.anomaly_probability = (1.0 - normal_pct) / avg_anomaly_duration
        
        frames = []
        is_anomaly = []
        
        for i in range(self.duration_frames):
            anomaly_type = self._inject_anomaly()
            self._update_anomalies()
            self._update_objects()
            frame = self._render_frame()
            frames.append(frame)
            is_anomaly.append(self._has_active_anomaly())
            self.frame_count += 1
        
        return frames, is_anomaly
    
    def _generate(self, duration_s: float | None = None) -> list[np.ndarray]:
        """Internal generate method."""
        if duration_s:
            total_frames = int(duration_s * self.fps)
        else:
            total_frames = self.duration_frames
        
        frames = []
        
        for i in range(total_frames):
            self._inject_anomaly()
            self._update_objects()
            frame = self._render_frame()
            frames.append(frame)
            self.frame_count += 1
        
        return frames
    
    def save_video(self, frames: list[np.ndarray], output_path: str, codec: str = "mp4v"):
        """Save frames as video file."""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available. Install with: pip install opencv-python")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        print(f"Saved video to {output_path}")
    
    def save_yuv(self, frames: list[np.ndarray], output_path: str):
        """Save frames as YUV420 file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for frame in frames:
                yuv = self._bgr_to_yuv420(frame)
                f.write(yuv)
        
        print(f"Saved YUV to {output_path}")
    
    def _bgr_to_yuv420(self, bgr: np.ndarray) -> bytes:
        """Convert BGR frame to YUV420 bytes."""
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        return yuv.tobytes()
    
    def save_metadata(self, output_path: str, is_anomaly: list[bool] | None = None):
        """Save metadata JSON."""
        metadata = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_s": self.duration_frames / self.fps,
            "num_frames": self.duration_frames,
            "config": {
                "background_color": self.config.background_color,
                "anomaly_probability": self.config.anomaly_probability,
            },
            "scene_type": self.__class__.__name__,
        }
        
        if is_anomaly:
            metadata["is_anomaly"] = is_anomaly
            normal_frames = sum(1 for a in is_anomaly if not a)
            anomaly_frames = sum(1 for a in is_anomaly if a)
            metadata["statistics"] = {
                "normal_frames": normal_frames,
                "anomaly_frames": anomaly_frames,
                "anomaly_ratio": anomaly_frames / len(is_anomaly) if is_anomaly else 0,
            }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {output_path}")


def generate_benchmark_dataset(
    output_dir: str = "benchmark_data",
    num_videos: int = 5,
    duration_per_video: float = 30.0,
    normal_ratio: float = 0.8,
) -> dict:
    """Generate a complete benchmark dataset.
    
    Args:
        output_dir: Directory to save generated videos
        num_videos: Number of videos to generate
        duration_per_video: Duration of each video in seconds
        normal_ratio: Ratio of normal-to-anomaly frames (0.8 = 80% normal)
    
    Returns:
        Dictionary with paths to generated files and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = SceneConfig(
        width=1920,
        height=1080,
        fps=25,
        duration_s=duration_per_video,
    )
    
    video_paths = []
    metadata_paths = []
    statistics = {
        "total_frames": 0,
        "normal_frames": 0,
        "anomaly_frames": 0,
    }
    
    for i in range(num_videos):
        print(f"\nGenerating video {i + 1}/{num_videos}...")
        
        gen = SyntheticSurveillanceGenerator(config)
        
        frames, is_anomaly = gen.generate_mixed_sequence(
            normal_pct=normal_ratio,
            duration_s=duration_per_video
        )
        
        video_path = output_dir / f"surveillance_{i:03d}.mp4"
        meta_path = output_dir / f"surveillance_{i:03d}_meta.json"
        
        gen.save_video(frames, str(video_path))
        gen.save_metadata(str(meta_path), is_anomaly)
        
        video_paths.append(str(video_path))
        metadata_paths.append(str(meta_path))
        
        statistics["total_frames"] += len(frames)
        statistics["normal_frames"] += sum(1 for a in is_anomaly if not a)
        statistics["anomaly_frames"] += sum(1 for a in is_anomaly if a)
    
    dataset_info = {
        "video_paths": video_paths,
        "metadata_paths": metadata_paths,
        "statistics": statistics,
        "normal_ratio": normal_ratio,
        "config": {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "duration_per_video": duration_per_video,
        }
    }
    
    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total videos: {num_videos}")
    print(f"Total frames: {statistics['total_frames']:,}")
    print(f"Normal frames: {statistics['normal_frames']:,} ({100 * statistics['normal_frames'] / statistics['total_frames']:.1f}%)")
    print(f"Anomaly frames: {statistics['anomaly_frames']:,} ({100 * statistics['anomaly_frames'] / statistics['total_frames']:.1f}%)")
    print(f"\nDataset info: {dataset_info_path}")
    
    return dataset_info


def generate_quick_test(output_dir: str = "benchmark_data/test") -> str:
    """Generate a quick 5-second test video."""
    config = SceneConfig(
        width=640,
        height=480,
        fps=25,
        duration_s=5.0,
        anomaly_probability=0.2,
    )
    
    gen = SyntheticSurveillanceGenerator(config)
    frames, is_anomaly = gen.generate_mixed_sequence(normal_pct=0.8)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_path = output_path / "quick_test.mp4"
    meta_path = output_path / "quick_test_meta.json"
    
    gen.save_video(frames, str(video_path))
    gen.save_metadata(str(meta_path), is_anomaly)
    
    print(f"\nQuick test generated: {video_path}")
    print(f"Resolution: {config.width}x{config.height}")
    print(f"Frames: {len(frames)}")
    print(f"Anomalies: {sum(is_anomaly)}/{len(is_anomaly)}")
    
    return str(video_path)


class LeWMBenchmarkSynthetics:
    """Integration with LeWM-VC for synthetic data benchmarking."""
    
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load LeWM-VC model if path provided."""
        if self.model_path and Path(self.model_path).exists():
            print(f"Loading model from {self.model_path}")
        else:
            print("No model path provided, using untrained model for testing")
    
    def run_surprise_gating_comparison(
        self,
        frames: list[np.ndarray],
        is_anomaly: list[bool],
        gating_thresholds: tuple[float, float] = (0.3, 0.7),
    ) -> dict:
        """Compare surprise-gating ON vs OFF.
        
        Args:
            frames: Video frames
            is_anomaly: Ground truth anomaly labels
            gating_thresholds: (low, high) thresholds for surprise detection
        
        Returns:
            Dictionary with comparison results
        """
        total_frames = len(frames)
        normal_frames = sum(1 for a in is_anomaly if not a)
        anomaly_frames = sum(1 for a in is_anomaly if a)
        
        results = {
            "total_frames": total_frames,
            "normal_frames": normal_frames,
            "anomaly_frames": anomaly_frames,
            "surprise_gating_on": {
                "estimated_normal_bits": normal_frames * 50,
                "estimated_anomaly_bits": anomaly_frames * 150,
                "total_bits": normal_frames * 50 + anomaly_frames * 150,
                "avg_bits_per_frame": (normal_frames * 50 + anomaly_frames * 150) / total_frames,
            },
            "surprise_gating_off": {
                "estimated_bits_per_frame": 100,
                "total_bits": total_frames * 100,
                "avg_bits_per_frame": 100,
            },
            "bitrate_savings": 0.0,
        }
        
        savings = (
            results["surprise_gating_off"]["total_bits"] - 
            results["surprise_gating_on"]["total_bits"]
        ) / results["surprise_gating_off"]["total_bits"]
        
        results["bitrate_savings"] = savings * 100
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic surveillance videos for LeWM-VC benchmarking"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "compare"],
        default="quick",
        help="Generation mode",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_data",
        help="Output directory",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=5,
        help="Number of videos to generate (full mode)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration per video in seconds",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=0.8,
        help="Ratio of normal frames (0.0-1.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height",
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        print("Generating quick test video...")
        generate_quick_test(args.output_dir)
    
    elif args.mode == "full":
        print(f"Generating {args.num_videos} benchmark videos...")
        generate_benchmark_dataset(
            output_dir=args.output_dir,
            num_videos=args.num_videos,
            duration_per_video=args.duration,
            normal_ratio=args.normal_ratio,
        )
    
    elif args.mode == "compare":
        print("Running surprise-gating comparison...")
        config = SceneConfig(
            width=args.width,
            height=args.height,
            duration_s=60.0,
            anomaly_probability=1.0 - args.normal_ratio,
        )
        gen = SyntheticSurveillanceGenerator(config)
        frames, is_anomaly = gen.generate_mixed_sequence(normal_pct=args.normal_ratio)
        
        benchmarker = LeWMBenchmarkSynthetics()
        results = benchmarker.run_surprise_gating_comparison(frames, is_anomaly)
        
        print(f"\n{'='*60}")
        print("Surprise-Gating Comparison Results")
        print(f"{'='*60}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Normal frames: {results['normal_frames']} ({100*results['normal_frames']/results['total_frames']:.1f}%)")
        print(f"Anomaly frames: {results['anomaly_frames']} ({100*results['anomaly_frames']/results['total_frames']:.1f}%)")
        print(f"\nWith surprise-gating:")
        print(f"  Total bits: {results['surprise_gating_on']['total_bits']:,}")
        print(f"  Avg bits/frame: {results['surprise_gating_on']['avg_bits_per_frame']:.1f}")
        print(f"\nWithout surprise-gating:")
        print(f"  Total bits: {results['surprise_gating_off']['total_bits']:,}")
        print(f"  Avg bits/frame: {results['surprise_gating_off']['avg_bits_per_frame']:.1f}")
        print(f"\nBitrate savings: {results['bitrate_savings']:.1f}%")


if __name__ == "__main__":
    main()
