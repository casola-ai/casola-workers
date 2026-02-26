#!/bin/bash

set -e

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/casola-ai/worker-vllm-omni}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CONTEXT="${SCRIPT_DIR}/.."

echo "Building Docker image: ${FULL_IMAGE}"
echo "  Dockerfile: ${SCRIPT_DIR}/Dockerfile"
echo "  Context:    ${BUILD_CONTEXT}"
docker build --platform linux/amd64 -f "${SCRIPT_DIR}/Dockerfile" -t "${FULL_IMAGE}" "${BUILD_CONTEXT}"

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
echo "To run locally:"
echo "  docker run -e CASOLA_API_URL=<url> -e CASOLA_API_TOKEN=<token> -e VLLM_MODEL=<model> ${FULL_IMAGE}"
