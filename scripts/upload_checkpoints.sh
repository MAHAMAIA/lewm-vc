#!/bin/bash
# upload_checkpoints.sh
# Uploads checkpoints from local machine to the droplet

# Update these variables for your setup
DROPLET_IP="165.245.135.111"
LOCAL_CHECKPOINTS="/Users/pm/Downloads/lewm_essentials/checkpoints_corrected"
REMOTE_DIR="/root/le-maia/checkpoints_rd_scratch"

echo "=== Upload Checkpoints to Droplet ==="
echo "Droplet IP: $DROPLET_IP"
echo "Local: $LOCAL_CHECKPOINTS"
echo "Remote: $REMOTE_DIR"
echo ""

# Create remote directory
ssh root@$DROPLET_IP "mkdir -p $REMOTE_DIR"

# Upload all checkpoint files
scp -r $LOCAL_CHECKPOINTS/* root@$DROPLET_IP:$REMOTE_DIR/

echo ""
echo "Upload complete!"
ssh root@$DROPLET_IP "ls -la $REMOTE_DIR"

