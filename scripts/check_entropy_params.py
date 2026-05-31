# check_entropy_params.py
import torch
import sys
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.entropy import HyperpriorEntropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
phase1_ckpt = '/root/le-maia/checkpoints/phase1_lambda_0.1/final.pt'
state = torch.load(phase1_ckpt, map_location=device)

# Check if 'entropy_model' key exists
if 'entropy_model' in state:
    entropy_model.load_state_dict(state['entropy_model'])
    print("Loaded entropy_model from checkpoint")
else:
    print("No 'entropy_model' key. Keys:", state.keys())

# Print some parameter statistics
for name, param in entropy_model.named_parameters():
    print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
