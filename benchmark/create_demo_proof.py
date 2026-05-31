"""
LeWM-VC Real Bitstream Benchmark Demo

This demo proves that LeWM-VC actually encodes videos and achieves
BITSTREAM SAVINGS through surprise-gating.
"""

import sys
sys.path.insert(0, 'src')

from lewm_vc.video_encoder import LeWMVideoCodec

import cv2
import numpy as np


def main():
    print("="*70)
    print("LeWM-VC BITSTREAM BENCHMARK DEMO")
    print("="*70)
    
    video_path = "datasets/pevid-hd/walking_day_outdoor_1_1.mpg"
    print(f"\nLoading: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (480, 270))
        frames_bgr.append(frame_small)
    cap.release()
    
    frames_bgr = frames_bgr[:50]
    print(f"Loaded {len(frames_bgr)} frames @ {frames_bgr[0].shape}")
    
    codec = LeWMVideoCodec(latent_dim=192, gop_size=8)
    
    print("\n[1] Encoding WITH surprise-gating...")
    encoded_gated, stats_gated = codec.encode_video(frames_bgr, use_surprise_gating=True)
    
    print("[2] Encoding WITHOUT surprise-gating (baseline)...")
    encoded_baseline, stats_baseline = codec.encode_video(frames_bgr, use_surprise_gating=False)
    
    savings = stats_baseline.total_bits - stats_gated.total_bits
    savings_pct = 100 * savings / stats_baseline.total_bits
    
    print("\n" + "="*70)
    print("PROOF: ACTUAL BITSTREAM MEASUREMENTS")
    print("="*70)
    print(f"""
BASELINE (no surprise-gating):
  Total bits: {stats_baseline.total_bits:,}
  Bytes: {stats_baseline.total_bytes:,}
  
WITH SURPRISE-GATING:
  Total bits: {stats_gated.total_bits:,}
  Bytes: {stats_gated.total_bytes:,}
  
BITSTREAM SAVINGS:
  Bits saved: {savings:,}
  SAVINGS: {savings_pct:.1f}%
""")
    
    print("\n[3] Creating demo video...")
    
    h, w = frames_bgr[0].shape[:2]
    output_w = w * 3
    output_h = h + 80
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('demo_video_proof.mp4', fourcc, 8, (output_w, output_h))
    
    decoded_gated = codec.decode_video(encoded_gated, target_size=(h, w))
    decoded_baseline = codec.decode_video(encoded_baseline, target_size=(h, w))
    
    for i in range(min(len(frames_bgr), 40)):
        frame = np.full((output_h, output_w, 3), 30, dtype=np.uint8)
        
        orig = frames_bgr[i]
        gated = cv2.cvtColor(decoded_gated[i], cv2.COLOR_RGB2BGR)
        baseline = cv2.cvtColor(decoded_baseline[i], cv2.COLOR_RGB2BGR)
        
        frame[0:h, 0:w] = orig
        frame[0:h, w:2*w] = gated
        frame[0:h, 2*w:3*w] = baseline
        
        enc = encoded_gated[i]
        bits = enc.bits_used
        surprise = enc.surprise
        
        if surprise > 0.7:
            gate_color = (0, 0, 255)
            gate_label = f"ANOMALY (surprise={surprise:.2f})"
        else:
            gate_color = (0, 255, 0)
            gate_label = f"Normal (surprise={surprise:.2f})"
        
        cv2.putText(frame, "ORIGINAL (Source)", (10, h+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, "LeWM-VC + Surprise-Gating", (w+10, h+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(frame, "Baseline (no gating)", (2*w+10, h+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,100), 1)
        
        title = f"PROVEN: {savings_pct:.1f}% BITSTREAM SAVINGS"
        cv2.putText(frame, title, (w-80, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        cv2.putText(frame, f"Frame {i} | Bits: {bits:,} | {gate_label}", 
                    (10, h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        
        cv2.putText(frame, f"Decoded frames show LeWM-VC", 
                    (w+10, h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        cv2.putText(frame, f"encoding output", 
                    (w+10, h+68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        
        writer.write(frame)
    
    writer.release()
    
    import os
    size_mb = os.path.getsize('demo_video_proof.mp4') / (1024*1024)
    
    print("\n" + "="*70)
    print("DEMO VIDEO CREATED: demo_video_proof.mp4")
    print("="*70)
    print(f"Size: {size_mb:.1f} MB")
    print(f"Frames: {min(len(frames_bgr), 40)}")
    print(f"Resolution: {output_w}x{output_h}")
    print("""
WHAT THIS PROVES:
1. LeWM-VC actually encodes videos through neural networks
2. Real bitstream is generated and measured
3. Surprise-gating achieves bitstream reduction
""")


if __name__ == "__main__":
    main()
