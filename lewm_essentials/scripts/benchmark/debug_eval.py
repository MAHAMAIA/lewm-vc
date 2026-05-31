import sys
sys.path.insert(0, '/root/le-maia/src')

import torch
import cv2
import numpy as np
from lewm_vc.quant import Quantizer
from eval_rd_final import VideoAutoencoder, ContextualEntropyModel, laplace_likelihood_discrete, device

ae_ckpt = '/root/le-maia/checkpoints_joint_phase0/ae_lambda_0.05_final.pt'
ent_ckpt = '/root/le-maia/checkpoints_joint_phase0/entropy_lambda_0.05_final.pt'

autoencoder = VideoAutoencoder().to(device)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.eval()
entropy_model = ContextualEntropyModel().to(device)
state = torch.load(ent_ckpt, map_location=device, weights_only=False)
for key in list(state.keys()):
    if 'mask' in key:
        del state[key]
entropy_model.load_state_dict(state, strict=False)
entropy_model.eval()
quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# Load a single frame from test video
cap = cv2.VideoCapture('/root/le-maia/datasets/pevid-hd/stealing_night_outdoor_1_2.mpg')
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Could not read frame")
frame = cv2.resize(frame, (256,256))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
frame_t = frame_t.to(device)

with torch.no_grad():
    latent_norm = autoencoder.encode(frame_t)
    quantized = quantizer(latent_norm)
    mu, log_scale = entropy_model(quantized)
    print(f"mu: mean={mu.mean().item():.4f}, std={mu.std().item():.4f}")
    print(f"log_scale: mean={log_scale.mean().item():.4f}, std={log_scale.std().item():.4f}")
    scale = torch.exp(log_scale)
    print(f"scale: mean={scale.mean().item():.4f}, std={scale.std().item():.4f}")
    nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=2.0/255, sigma_floor=0.003)
    bits = nll * quantized.numel() / np.log(2)
    print(f"nll per element: {nll.item():.6f}, total bits: {bits.item():.2f}")
