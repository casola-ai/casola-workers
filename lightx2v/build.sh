#!/bin/bash

set -e

IMAGE_NAME="${IMAGE_NAME:-registry.casola-staging.net/casola-ai/worker-lightx2v}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Build from workers/ root so Dockerfile can access casola_worker/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}/.."

echo "Building Docker image: ${FULL_IMAGE}"
docker build --platform linux/amd64 -t "${FULL_IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${BUILD_CONTEXT}"

echo "Pushing Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Build complete!"
echo "Image: ${FULL_IMAGE}"
echo ""
echo "To run locally:"
echo "  docker run --gpus all \\"
echo "    -e CASOLA_WS_URL=ws://localhost:8787 \\"
echo "    -e CASOLA_CONFIG_ID=test \\"
echo "    -e LIGHTX2V_MODEL_PATH=/models/Wan2.1-T2V-1.3B \\"
echo "    -e LIGHTX2V_MODEL_CLS=wan2.1 \\"
echo "    -e LIGHTX2V_CONFIG_JSON=/app/configs/wan/wan_t2v.json \\"
echo "    ${FULL_IMAGE}"
