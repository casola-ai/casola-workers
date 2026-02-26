#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="${BASE_IMAGE:-ghcr.io/casola-ai/worker-comfy-base}"
BASE_TAG="${BASE_TAG:-latest}"

echo "Building base image: ${BASE_IMAGE}:${BASE_TAG}"
docker build --platform linux/amd64 -t "${BASE_IMAGE}:${BASE_TAG}" -f base/Dockerfile ..

# Build all workflows, also tag as :latest
for workflow_dir in workflows/*/; do
    workflow_name=$(basename "$workflow_dir")
    IMAGE="${WORKFLOW_IMAGE:-ghcr.io/casola-ai/worker-comfy}"
    TAG="${workflow_name}"
    echo "Building workflow image: ${IMAGE}:${TAG}"
    docker build --platform linux/amd64 -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" "$workflow_dir"
done

echo ""
echo "Pushing images..."
docker push "${BASE_IMAGE}:${BASE_TAG}"
for workflow_dir in workflows/*/; do
    workflow_name=$(basename "$workflow_dir")
    IMAGE="${WORKFLOW_IMAGE:-ghcr.io/casola-ai/worker-comfy}"
    docker push "${IMAGE}:${workflow_name}"
    docker push "${IMAGE}:latest"
done

echo ""
echo "Build and push complete!"
echo ""
echo "To run locally (requires GPU):"
echo "  docker run --gpus all \\"
echo "    -e CASOLA_API_URL=<url> \\"
echo "    -e CASOLA_API_TOKEN=<token> \\"
echo "    -e CASOLA_QUEUE_ID=<queue> \\"
echo "    -e CASOLA_CONFIG_ID=<config-id> \\"
echo "    ghcr.io/casola-ai/worker-comfy:qwen-image-2512"
