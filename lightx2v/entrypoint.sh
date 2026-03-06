#!/bin/bash
# LightX2V worker entrypoint — thin wrapper around shared entrypoint.
export WORKER_NAME="LightX2V"
exec "$(dirname "$0")/shared/entrypoint.sh"
