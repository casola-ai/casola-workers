#!/bin/bash

set -e

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/casola-ai/worker-vllm}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Build from workers/ root so Dockerfile can access casola_worker/
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building Docker image: ${FULL_IMAGE}"
docker build --platform linux/amd64 -t "${FULL_IMAGE}" -f "${REPO_ROOT}/vllm/Dockerfile" "${REPO_ROOT}"

echo "Pushing Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

# Build combo image with baked-in model weights if MODEL_IMAGE is set
if [ -n "${MODEL_IMAGE:-}" ]; then
  MODEL_LABEL="${MODEL_LABEL:-${MODEL_IMAGE##*/}}"
  MODEL_LABEL="${MODEL_LABEL%%:*}"
  COMBO_IMAGE="${IMAGE_NAME}:${MODEL_LABEL}"
  echo ""
  echo "Building combo image: ${COMBO_IMAGE}"
  "${REPO_ROOT}/tools/combine.sh" "${FULL_IMAGE}" "${MODEL_IMAGE}" "${COMBO_IMAGE}"
fi

echo ""
echo "Build complete!"
echo "Image: ${FULL_IMAGE}"
echo ""
echo "To push to registry:"
echo "  docker push ${FULL_IMAGE}"
echo ""
echo "To run locally:"
echo "  docker run -e CASOLA_API_URL=<url> -e CASOLA_API_TOKEN=<token> -e VLLM_MODEL=<model> ${FULL_IMAGE}"
