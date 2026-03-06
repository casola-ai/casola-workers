#!/bin/bash

set -e

IMAGE_NAME="${IMAGE_NAME:-registry.casola-staging.net/casola-ai/worker-vllm}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Build from workers/ root so Dockerfile can access casola_worker/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}/.."

echo "Building Docker image: ${FULL_IMAGE}"
docker build --platform linux/amd64 -t "${FULL_IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${BUILD_CONTEXT}"

echo "Pushing Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

# Build combo image with baked-in model weights if MODEL_IMAGE is set
if [ -n "${MODEL_IMAGE:-}" ]; then
  MODEL_LABEL="${MODEL_LABEL:-${MODEL_IMAGE##*/}}"
  MODEL_LABEL="${MODEL_LABEL%%:*}"
  COMBO_IMAGE="${IMAGE_NAME}:${MODEL_LABEL}"
  echo ""
  echo "Building combo image: ${COMBO_IMAGE}"
  "${BUILD_CONTEXT}/tools/combine.sh" "${FULL_IMAGE}" "${MODEL_IMAGE}" "${COMBO_IMAGE}"
fi

echo ""
echo "Build complete!"
echo "Image: ${FULL_IMAGE}"
echo ""
echo "To upload to R2 registry (CI path):"
echo "  workers/tools/r2-upload.sh ./image-oci casola-ai/worker-vllm \$TAG"
echo ""
echo "To run locally:"
echo "  docker run -e CASOLA_API_URL=<url> -e CASOLA_API_TOKEN=<token> -e VLLM_MODEL=<model> ${FULL_IMAGE}"
