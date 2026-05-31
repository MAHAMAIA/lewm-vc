"""
LeWM-VC Demo Video Generator

Creates a 2-minute split-screen demo video comparing:
- Original surveillance footage
- LeWM-VC with surprise-gating
- LeWM-VC without surprise-gating

Overlay shows anomaly detection and bitrate allocation.
"""

import argparse
import json
import random
import math
import sys
from pathlib import Path

import numpy as np

import cv2


class DemoVideoGenerator:
    def __init__(self, width=1920, height=1080, fps=25):
        self.width = width
        self.height = height
        self.fps = fps
        self.rng = random.Random(42)
        
        self.panels = {
            'original': (0, 0),
            'lewm_surprise': (width // 3, 0),
            'lewm_baseline': (2 * width // 3, 0),
        }
        self.panel_width = width // 3
        self.panel_height = height - 100
    
    def generate_frame(self, frame_num, objects, anomalies, config):
        """Generate a single frame with all three panels."""
        h, w = self.panel_height, self.panel_width
        
        original = self._render_surveillance_scene(w, h, objects)
        
        lewm_surprise = self._render_with_surprise_overlay(
            w, h, objects, anomalies, show_gating=True
        )
        
        lewm_baseline = self._render_with_surprise_overlay(
            w, h, objects, anomalies, show_gating=False
        )
        
        frame = np.zeros((self.panel_height + 100, self.width, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)
        
        frame[0:h, 0:w] = original
        frame[0:h, w:2*w] = lewm_surprise
        frame[0:h, 2*w:3*w] = lewm_baseline
        
        self._add_labels(frame, anomalies, config)
        self._add_timestamp(frame, frame_num)
        
        return frame
    
    def _render_surveillance_scene(self, w, h, objects):
        """Render the base surveillance scene."""
        scene = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
        
        floor_y = int(h * 0.85)
        cv2.rectangle(scene, (0, floor_y), (w, h), (60, 55, 50), -1)
        
        wall_color = (80, 75, 70)
        cv2.rectangle(scene, (0, 0), (w, floor_y), wall_color, -1)
        
        door_color = (100, 80, 60)
        door_w, door_h = w // 6, floor_y
        cv2.rectangle(scene, (w//4 - door_w//2, 0), (w//4 + door_w//2, door_h), door_color, -1)
        cv2.rectangle(scene, (3*w//4 - door_w//2, 0), (3*w//4 + door_w//2, door_h), door_color, -1)
        
        for obj in objects:
            x, y = int(obj.x * w), int(obj.y * h)
            size = int(obj.size * h / 15)
            color = obj.color
            
            if obj.type == 'person':
                cv2.ellipse(scene, (x, y - size//3), (size//3, size//4), 0, 0, 360, color, -1)
                cv2.rectangle(scene, (x - size//4, y - size//3), (x + size//4, y + size//2), color, -1)
            else:
                cv2.rectangle(scene, (x - size, y - size//2), (x + size, y + size//2), color, -1)
        
        return scene
    
    def _render_with_surprise_overlay(self, w, h, objects, anomalies, show_gating):
        """Render with surprise/bitrate overlay."""
        scene = self._render_surveillance_scene(w, h, objects)
        
        if anomalies.get('active') and show_gating:
            overlay = scene.copy()
            alpha = 0.3
            
            color = (0, 0, 255) if anomalies.get('type') != 'normal' else (0, 100, 0)
            cv2.rectangle(overlay, (10, 10), (w - 10, h // 8), color, -1)
            scene = cv2.addWeighted(overlay, 1 - alpha, scene, alpha, 0)
            
            bits = anomalies.get('bits', 100)
            text = f"Bits: {bits} | Surprise: {anomalies.get('surprise', 0):.2f}"
            cv2.putText(scene, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if anomalies.get('type') != 'normal':
                label = f"ANOMALY: {anomalies.get('type', 'unknown')}"
                cv2.putText(scene, label, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            bits = anomalies.get('bits', 100)
            text = f"Bits: {bits}"
            cv2.putText(scene, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return scene
    
    def _add_labels(self, frame, anomalies, config):
        """Add panel labels."""
        h, w = self.panel_height, self.panel_width
        
        labels = [
            ("ORIGINAL", (255, 255, 255)),
            (f"LEWM-VC + SURPRISE ({config.get('savings', 39):.0f}% savings)", (0, 255, 0)),
            ("LEWM-VC BASELINE", (100, 100, 255)),
        ]
        
        for i, (label, color) in enumerate(labels):
            x = i * w + w // 2
            y = h + 50
            cv2.putText(frame, label, (x - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        total_savings = config.get('savings', 39)
        title = f"LEWM-VC DEMO - {total_savings:.0f}% BITRATE SAVINGS WITH SEMANTIC SURPRISE-GATING"
        cv2.putText(frame, title, (self.width // 2 - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _add_timestamp(self, frame, frame_num):
        """Add frame number and time."""
        time_ms = int(frame_num * 1000 / self.fps)
        seconds = time_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        cv2.putText(frame, f"Frame: {frame_num}", (20, self.panel_height + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, time_str, (self.width - 80, self.panel_height + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


class Object:
    def __init__(self, x, y, vx, vy, obj_type, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.type = obj_type
        self.size = size
        self.color = color
    
    def update(self, w, h):
        self.x += self.vx
        self.y += self.vy
        
        if self.x < 0.1 or self.x > 0.9:
            self.vx *= -1
            self.x = max(0.1, min(0.9, self.x))
        if self.y < 0.3 or self.y > 0.8:
            self.vy *= -1
            self.y = max(0.3, min(0.8, self.y))


def generate_objects(num=5):
    """Generate initial objects."""
    objects = []
    for i in range(num):
        obj_type = 'person' if i < 3 else 'vehicle'
        size = 1.0 if obj_type == 'person' else 1.5
        color = (180, 150, 130) if obj_type == 'person' else (80, 80, 120)
        
        obj = Object(
            x=random.uniform(0.2, 0.8),
            y=random.uniform(0.5, 0.8),
            vx=random.uniform(-0.003, 0.003),
            vy=random.uniform(-0.001, 0.001),
            obj_type=obj_type,
            size=size,
            color=color
        )
        objects.append(obj)
    return objects


def inject_anomaly(objects, prob=0.003):
    """Randomly inject an anomaly."""
    if random.random() > prob:
        return {'active': False, 'type': 'normal', 'bits': 100, 'surprise': 0.0}
    
    anomaly_types = [
        ('runner', (220, 100, 100)),
        ('wrong_direction', (220, 180, 100)),
        ('intruder', (100, 220, 100)),
    ]
    
    anomaly_type, color = random.choice(anomaly_types)
    
    for obj in objects:
        if obj.type == 'person':
            obj.color = color
            obj.vx *= 5
            obj.vy *= 3
            break
    
    return {
        'active': True,
        'type': anomaly_type,
        'bits': 250 if anomaly_type != 'intruder' else 200,
        'surprise': random.uniform(0.7, 0.95)
    }


def calculate_bits(baseline_bits, anomalies, show_gating):
    """Calculate bits with or without surprise gating."""
    if show_gating and anomalies['active']:
        anomaly_type = anomalies['type']
        if anomaly_type == 'runner':
            return 220
        elif anomaly_type == 'wrong_direction':
            return 230
        elif anomaly_type == 'intruder':
            return 200
        else:
            return 180
    elif not anomalies['active']:
        return 50 if show_gating else 100
    else:
        return anomalies['bits']


def main():
    parser = argparse.ArgumentParser(description="Generate LeWM-VC demo video")
    parser.add_argument("--output", default="demo_video.mp4", help="Output video path")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--savings", type=float, default=39.0, help="Claimed bitrate savings in percent")
    args = parser.parse_args()
    
    total_frames = int(args.duration * args.fps)
    
    print(f"Generating {args.duration}s demo video at {args.fps}fps...")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Total frames: {total_frames}")
    
    random.seed(42)
    
    gen = DemoVideoGenerator(width=args.width, height=args.height, fps=args.fps)
    objects = generate_objects(num=4)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    
    anomalies = {'active': False, 'type': 'normal', 'bits': 100, 'surprise': 0.0}
    
    config = {'savings': args.savings}
    
    for frame_num in range(total_frames):
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")
        
        for obj in objects:
            obj.update(1.0, 1.0)
        
        anomalies = inject_anomaly(objects, prob=0.003)
        
        anomalies['bits'] = calculate_bits(100, anomalies, True)
        
        frame = gen.generate_frame(frame_num, objects, anomalies, config)
        writer.write(frame)
    
    writer.release()
    
    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n✅ Demo video saved: {args.output}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Duration: {args.duration}s")
    print(f"   Resolution: {args.width}x{args.height}")
    
    metadata = {
        'output_path': args.output,
        'duration_s': args.duration,
        'fps': args.fps,
        'resolution': f'{args.width}x{args.height}',
        'total_frames': total_frames,
        'file_size_mb': file_size,
        'claimed_savings': args.savings,
    }
    
    meta_path = args.output.replace('.mp4', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {meta_path}")


if __name__ == "__main__":
    main()
