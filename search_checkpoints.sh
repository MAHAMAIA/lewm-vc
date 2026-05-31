#!/bin/bash
# Run from /Users/pm/Documents/dev/le-maia

echo "=== Checking checkpoint directories ==="
for d in checkpoints_corrected checkpoints_rd_scratch checkpoints_gmm checkpoints_amd_final checkpoints_jepa checkpoints_joint_phase0; do
    if [ -d "$d" ]; then
        count=$(ls -1 "$d"/*.pt 2>/dev/null | wc -l)
        echo "$d: $count .pt files"
        ls -lh "$d"/*.pt 2>/dev/null | head -5
    else
        echo "$d: DOES NOT EXIST"
    fi
    echo ""
done

echo "=== Checking for training logs ==="
find . -maxdepth 3 \( -name "*.log" -o -name "nohup.out" -o -name "slurm-*.out" \) -ls 2>/dev/null

echo "=== Checking GPU ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')" 2>/dev/null
