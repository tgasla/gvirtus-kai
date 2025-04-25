#!/usr/bin/env bash
set -euo pipefail

NODE_NAME="${NODE_NAME:-}"

# Loop to run the script every minute
while true; do
  # 1) Is gvirtus_backend running on host?
  if pgrep -f gvirtus-backend >/dev/null; then
    kubectl label node "$NODE_NAME" \
      gvirtus-role=backend gpu-virtualization=none \
      --overwrite && echo "BACKEND"
  else
    # 2) Otherwise, mark as frontend
    kubectl label node "$NODE_NAME" \
      gvirtus-role=frontend gpu-virtualization=vgpu \
      --overwrite && echo "FRONTEND"
  fi

  # 3) (Optional) choose best backend
  # … your logic here …

  # Sleep for 60 seconds before checking again
  sleep 60
done
