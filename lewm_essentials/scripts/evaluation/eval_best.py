import sys
sys.path.insert(0, '/root/le-maia/src')
import torch
import cv2
import glob
import numpy as np
from tqdm import tqdm
from lewm_vc.quant import Quantizer
from train_joint_phase0 import VideoAutoencoder, ContextualEntropyModel, laplace_likelihood_discrete, QUANT_STEP, device

checkpoint_dir = '/root/le-maia/checkpoints_joint_phase0'
lambda_list = [0.05, 0.1, 0.5, 1.0, 5.0]  # 0.01 missing
target_size = (256,256)

test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
cap = cv2.VideoCapture(test_video)
frames = []
while len(frames) < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)

for lam in lambda_list:
    ae_ckpt = os.path.join(checkpoint_dir, f'ae_lambda_{lam}_best.pt')
    ent_ckpt = os.path.join(checkpoint_dir, f'entropy_lambda_{lam}_best.pt')
    if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
        print(f"Missing best checkpoints for λ={lam}, skipping")
        continue
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
    total_bits = 0
    total_mse = 0
    for frame in tqdm(frames, desc=f"λ={lam}"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = autoencoder.encode(frame_t)
            quantized = quantizer(latent_norm)
            mu, log_scale = entropy_model(quantized)
            # Print first frame stats
            if total_bits == 0:
                print(f"mu mean={mu.mean().item():.4f}, std={mu.std().item():.4f}")
                print(f"log_scale mean={log_scale.mean().item():.4f}, std={log_scale.std().item():.4f}")
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=2.0/255, sigma_floor=0.003)
            bits = nll * quantized.numel() / np.log(2)
            total_bits += bits.item()
            recon = autoencoder.decode(quantized, target_size=target_size)
            mse = torch.nn.functional.mse_loss(recon, frame_t).item()
            total_mse += mse
    bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
    psnr = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
    print(f"λ={lam}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")
