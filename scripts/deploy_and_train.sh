#!/usr/bin/env bash
# Deploy LeWM-VC to training droplet and launch training.
#
# Modes:
#   --fresh         First-time setup: copy code + install deps + launch
#   --train         Copy code + install deps + validate + launch (default)
#   --validate      Copy code + validate only
#   --monitor       Tail latest log
#   --attach        Attach to tmux session
#
# Profiles:
#   single  1x MI300X — Phase 0 warmup (default)
#   x8      8x MI300X — all 5 lambdas on GPUs 0-4
#
# Env overrides:
#   REMOTE_HOST     (default: from deploy_config.sh)
#   REMOTE_USER     (default: root)
#   DOCKER_NAME     (default: rocm — set to empty for bare metal)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Config ----
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/deploy_config.sh}"
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST in deploy_config.sh}"
REMOTE_PATH="${REMOTE_PATH:-/root/le-maia}"
DOCKER_NAME="${DOCKER_NAME:-rocm}"
CONTAINER_PATH="/workspace/le-maia"
PROFILE="${PROFILE:-single}"
MODE="${1:-train}"
DRY_RUN=false

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh)  MODE="fresh" ;;
        --train)  MODE="train" ;;
        --validate) MODE="validate" ;;
        --monitor) MODE="monitor" ;;
        --attach)  MODE="attach" ;;
        --dry-run) DRY_RUN=true ;;
        --profile) shift; PROFILE="$1" ;;
        --config) shift; CONFIG_FILE="$1"; [ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE" ;;
        *) ;;
    esac
    shift
done

RSYNC_DEST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
SSH_CMD="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 ${REMOTE_USER}@${REMOTE_HOST}"
TMUX_SESSION="lewm_${PROFILE}"
LOG_DIR="${REMOTE_PATH}/logs"
DOCKER_WORKDIR="${CONTAINER_PATH}"
DOCKER="${DOCKER_NAME:+docker exec -w ${DOCKER_WORKDIR} ${DOCKER_NAME}}"

case "$PROFILE" in
    single) echo "[cost] ~\$1.99/hr (1x MI300X)" ;;
    x8)
        echo ""
        echo "  x8 profile = 8x MI300X = ~\$15.92/hr"
        echo "  Consider: Phase 0-2 on single (\$1.99/hr)"
        echo "  Only use x8 for multi-lambda Phase 3+"
        echo ""
        ;;
esac
echo "[host] ${REMOTE_USER}@${REMOTE_HOST}  [docker] ${DOCKER_NAME:-bare-metal}"

run() { if [ "$DRY_RUN" = true ]; then echo "[dry-run] $*"; else echo "[run] $*"; eval "$*"; fi; }
rem() { if [ "$DRY_RUN" = true ]; then echo "[dry-run-ssh] $*"; else $SSH_CMD "$*"; fi; }
dexec() { rem "${DOCKER} $*"; }

# ---- 1. Rsync to host ----
rsync_project() {
    echo "--- Rsync project ---"
    rsync -az --delete \
        -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3" \
        --include='src/' --include='src/**' \
        --include='scripts/' --include='scripts/**' --include='scripts/training/' --include='scripts/training/**' \
        --include='configs/' --include='configs/**' \
        --include='tests/' --include='tests/**' \
        --include='pyproject.toml' --include='setup.py' --include='setup.cfg' \
        --include='README.md' --include='LICENSE' \
        --exclude='*' \
        "$PROJECT_DIR/" "$RSYNC_DEST/"
}

# ---- 2. Copy code into container ----
container_cp() {
    [ -z "$DOCKER_NAME" ] && return
    echo "--- Copy code into container (excluding datasets/checkpoints) ---"
    rem "docker exec ${DOCKER_NAME} mkdir -p ${CONTAINER_PATH} && tar --exclude=datasets --exclude=checkpoints -C ${REMOTE_PATH} -cf - . | docker exec -i ${DOCKER_NAME} tar -C ${CONTAINER_PATH} -xf -"
}

# ---- 2b. Copy datasets into container ----
container_cp_datasets() {
    [ -z "$DOCKER_NAME" ] && return
    echo "--- Copy datasets into container ---"
    local ds="$1"
    rem "docker exec ${DOCKER_NAME} mkdir -p ${CONTAINER_PATH}/datasets && tar -C ${REMOTE_PATH}/datasets -cf - ${ds} | docker exec -i ${DOCKER_NAME} tar -C ${CONTAINER_PATH}/datasets -xf -"
}

# ---- 3. Install deps inside container ----
install_deps() {
    echo "--- Install deps ---"
    dexec "pip install -e '${CONTAINER_PATH}[dev,train,wandb]' 2>&1 | tail -5"
    echo "[verify]"
    dexec "python3 -c \"import torch; v=torch.__version__; h=torch.version.hip; c=torch.cuda.is_available(); print(f'torch {v}, ROCm: {h}, CUDA: {c}')\""
}

# ---- 4. Validate ----
validate_dataset() {
    echo "--- Validate dataset ---"
    dexec "python3 ${CONTAINER_PATH}/scripts/validate_dataset.py --output ${CONTAINER_PATH}/validation_output"
}

# ---- 5. Launch Phase 0 ----
launch_phase0_single() {
    echo "--- Launch Phase 0 (single GPU) ---"
    rem "mkdir -p ${LOG_DIR}"
    local logfile="${LOG_DIR}/phase0_$(date +%Y%m%d_%H%M%S).log"
    local train_cmd="${DOCKER} python3 ${CONTAINER_PATH}/scripts/train_production.py --config ${CONTAINER_PATH}/configs/train_config.yaml --phase 0 --lambda 0.05 --wandb > ${logfile} 2>&1"
    rem "tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true"
    rem "tmux new-session -d -s ${TMUX_SESSION} '${train_cmd}'"
    echo "[done] tmux '${TMUX_SESSION}' started"
    echo "       Attach:  ./scripts/deploy_and_train.sh --attach"
    echo "       Monitor: ./scripts/deploy_and_train.sh --monitor"
}

# ---- 6. Monitor ----
monitor() {
    local pattern="phase0"
    local latest_log
    latest_log=$(rem "ls -t ${LOG_DIR}/${pattern}_*.log 2>/dev/null | head -1" | tr -d '\r')
    [ -z "$latest_log" ] && { echo "[error] no logs in ${LOG_DIR}/"; exit 1; }
    echo "Tailing: ${latest_log}"
    exec $SSH_CMD "tail -f ${latest_log}"
}

# ---- 7. Attach ----
attach_tmux() {
    exec $SSH_CMD -t "tmux attach -t ${TMUX_SESSION}"
}

# ---- Main ----
case "$MODE" in
    fresh)
        rsync_project
        container_cp
        install_deps
        validate_dataset
        [ "$PROFILE" = "x8" ] && launch_all_lambdas_x8 || launch_phase0_single
        ;;
    train)
        rsync_project
        container_cp
        container_cp_datasets "virat"
        install_deps
        validate_dataset
        [ "$PROFILE" = "x8" ] && launch_all_lambdas_x8 || launch_phase0_single
        ;;
    validate)
        rsync_project
        container_cp
        container_cp_datasets "virat"
        install_deps
        validate_dataset
        ;;
    monitor) monitor ;;
    attach) attach_tmux ;;
esac
