#!/usr/bin/env bash
# Sync changed code to the training droplet, skipping datasets/checkpoints/etc.
#
# Usage:
#   bash scripts/sync_to_droplet.sh <IP>                       # first time
#   bash scripts/sync_to_droplet.sh                            # reuse saved IP
#   bash scripts/sync_to_droplet.sh --push <IP>                # commit+push then sync
#   bash scripts/sync_to_droplet.sh --push "message" <IP>      # custom commit message
#
# Notes:
#   - Deletes stale .pyc/__pycache__ on the droplet after sync
#   - Preserves datasets/, checkpoints/ on the droplet
#   - Run from anywhere inside the repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HISTORY_FILE="$PROJECT_DIR/.droplet_ip"

# Source deploy config if it exists (REMOTE_HOST, REMOTE_USER)
CONFIG_FILE="$SCRIPT_DIR/deploy_config.sh"
[ -f "$CONFIG_FILE" ] && source "$CONFIG_FILE"
REMOTE_USER="${REMOTE_USER:-root}"

# ── Parse args ──
PUSH=false
COMMIT_MSG=""
IP=""

while [ $# -gt 0 ]; do
  case "$1" in
    --push)
      PUSH=true
      shift
      # Next arg is either a commit message or the IP
      if [ $# -gt 0 ] && ! [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        COMMIT_MSG="$1"
        shift
      fi
      ;;
    *)
      IP="$1"
      shift
      ;;
  esac
done

# ── Resolve IP (CLI arg > .droplet_ip > deploy_config REMOTE_HOST) ──
if [ -z "$IP" ]; then
  if [ -f "$HISTORY_FILE" ]; then
    IP="$(cat "$HISTORY_FILE")"
    echo "Using saved IP: $IP"
  elif [ -n "${REMOTE_HOST:-}" ]; then
    IP="$REMOTE_HOST"
    echo "Using REMOTE_HOST from config: $IP"
  else
    echo "Usage: bash scripts/sync_to_droplet.sh [--push \"message\"] <IP>"
    echo "  or save IP: echo '1.2.3.4' > .droplet_ip"
    echo "  or set REMOTE_HOST in scripts/deploy_config.sh"
    exit 1
  fi
else
  echo "$IP" > "$HISTORY_FILE"
fi

# ── Commit + push (if --push) ──
if [ "$PUSH" = true ]; then
  cd "$PROJECT_DIR"
  if [ -z "$(git status --porcelain)" ]; then
    echo "Nothing to commit, working tree clean."
  else
    if [ -z "$COMMIT_MSG" ]; then
      COMMIT_MSG="sync: $(date '+%Y-%m-%d %H:%M')"
    fi
    echo "Committing changes..."
    git add -A
    git commit -m "$COMMIT_MSG"
    echo "Pushing to origin..."
    git push origin 2>&1 | tail -3
  fi
fi

# ── Sync to droplet ──
REMOTE_PATH="${REMOTE_PATH:-/root/le-maia}"
REMOTE_DEST="${REMOTE_USER}@${IP}:${REMOTE_PATH}"
echo "Syncing $PROJECT_DIR -> ${REMOTE_DEST} ..."
# Whitelist: only sync what the droplet needs for training
rsync -avz --delete \
  --include='src/' --include='src/**' \
  --include='scripts/' --include='scripts/**' --include='scripts/training/' --include='scripts/training/**' \
  --include='configs/' --include='configs/**' \
  --include='tests/' --include='tests/**' \
  --include='pyproject.toml' --include='setup.py' --include='setup.cfg' \
  --include='README.md' --include='LICENSE' \
  --exclude='*' \
  -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3" \
  "$PROJECT_DIR"/ "${REMOTE_DEST}"

echo ""
echo "Cleaning __pycache__ on droplet..."
ssh -o ServerAliveInterval=15 "${REMOTE_USER}@${IP}" "find ${REMOTE_PATH} -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true"

# ── Copy into Docker container (if DOCKER_NAME is set) ──
if [ -n "${DOCKER_NAME:-}" ]; then
  echo ""
  echo "Copying code into container ${DOCKER_NAME} (excluding datasets)..."

  ssh -o ServerAliveInterval=15 "${REMOTE_USER}@${IP}" "docker exec ${DOCKER_NAME} mkdir -p /workspace/le-maia && \
    tar --exclude='datasets' --exclude='checkpoints' -C ${REMOTE_PATH} -cf - . | \
    docker exec -i ${DOCKER_NAME} tar -C /workspace/le-maia -xf -"
fi

echo ""
echo "Done. Attach with: ssh ${REMOTE_USER}@${IP} -t 'tmux attach || tmux new -s train'"
