#!/bin/bash
# vLLM worker entrypoint — thin wrapper around shared entrypoint.
export WORKER_NAME="vLLM"
exec "$(dirname "$0")/shared/entrypoint.sh"
