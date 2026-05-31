import torch
ae = torch.load('/root/le-maia/checkpoints_corrected/ae_lambda_0.01_final.pt', map_location='cpu')
ent = torch.load('/root/le-maia/checkpoints_corrected/entropy_lambda_0.01_final.pt', map_location='cpu')

# Check for key parameters
print("Autoencoder keys:", list(ae.keys())[:5])
print("Entropy model keys:", list(ent.keys())[:5])

# Check if decoder has 6 layers (for λ=0.01)
decoder_keys = [k for k in ae.keys() if 'decoder' in k]
print(f"Decoder layers: {len([k for k in decoder_keys if 'up' in k])}")
