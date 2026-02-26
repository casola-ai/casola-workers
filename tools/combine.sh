#!/usr/bin/env bash
# Combine a base worker image with a model image via registry-level blob mounting.
# The model layer is cross-mounted server-side — no large data transfers.
# Only a tiny config blob (~few KB) and manifest (~1 KB) are pushed.
#
# Usage: workers/combine.sh <base_image> <model_image> <combo_tag>
#
# Requires: crane, curl, jq, sha256sum (or shasum on macOS)
# Auth: reads credentials from ~/.docker/config.json (set by docker/login-action)
# Registry: GHCR-specific (uses ghcr.io token endpoint)
set -euo pipefail

BASE_IMAGE="${1:?Usage: combine.sh <base_image> <model_image> <combo_tag>}"
MODEL_IMAGE="${2:?Usage: combine.sh <base_image> <model_image> <combo_tag>}"
COMBO_TAG="${3:?Usage: combine.sh <base_image> <model_image> <combo_tag>}"

PLATFORM="linux/amd64"
REGISTRY="ghcr.io"

# --- Helpers ---

# Extract repo from ghcr.io/org/repo:tag -> org/repo
image_repo() { local r="${1#${REGISTRY}/}"; echo "${r%%[:@]*}"; }
# Extract tag from ghcr.io/org/repo:tag -> tag
image_tag() { local r="${1#${REGISTRY}/}"; local t="${r##*:}"; [ "$t" = "$r" ] && echo "latest" || echo "$t"; }

# Portable sha256
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Get a GHCR bearer token with the given scopes (using docker config credentials)
ghcr_token() {
  local creds user pass
  creds=$(echo "$REGISTRY" | crane auth get 2>/dev/null) || {
    echo "ERROR: failed to get credentials for ${REGISTRY}" >&2; exit 1
  }
  user=$(echo "$creds" | jq -r '.Username')
  pass=$(echo "$creds" | jq -r '.Secret')
  curl -s -u "${user}:${pass}" \
    "https://${REGISTRY}/token?service=${REGISTRY}&$(printf 'scope=%s&' "$@" | sed 's/&$//')" \
    | jq -r '.token'
}

TARGET_REPO=$(image_repo "$COMBO_TAG")
TARGET_TAG=$(image_tag "$COMBO_TAG")
MODEL_REPO=$(image_repo "$MODEL_IMAGE")

# Token with pull on model repo + pull/push on target repo
TOKEN=$(ghcr_token \
  "repository:${TARGET_REPO}:pull,push" \
  "repository:${MODEL_REPO}:pull")

AUTH=("Authorization: Bearer ${TOKEN}")

# --- 1. Read model image metadata ---
echo "Reading model image metadata..."
MODEL_MANIFEST=$(crane manifest --platform "$PLATFORM" "$MODEL_IMAGE")
LAYER_COUNT=$(echo "$MODEL_MANIFEST" | jq '.layers | length')
if [ "$LAYER_COUNT" -ne 1 ]; then
  echo "ERROR: model image has ${LAYER_COUNT} layers, expected 1" >&2; exit 1
fi

MODEL_LAYER_DIGEST=$(echo "$MODEL_MANIFEST" | jq -r '.layers[0].digest')
MODEL_LAYER_SIZE=$(echo "$MODEL_MANIFEST" | jq -r '.layers[0].size')
MODEL_LAYER_MEDIA=$(echo "$MODEL_MANIFEST" | jq -r '.layers[0].mediaType')
MODEL_DIFF_ID=$(crane config --platform "$PLATFORM" "$MODEL_IMAGE" | jq -r '.rootfs.diff_ids[0]')
echo "  Layer: ${MODEL_LAYER_DIGEST} (${MODEL_LAYER_SIZE} bytes)"

# --- 2. Read base image metadata ---
echo "Reading base image metadata..."
BASE_MANIFEST=$(crane manifest --platform "$PLATFORM" "$BASE_IMAGE")
BASE_CONFIG=$(crane config --platform "$PLATFORM" "$BASE_IMAGE")
echo "  Layers: $(echo "$BASE_MANIFEST" | jq '.layers | length')"

# --- 3. Cross-mount model blob (server-side, no data transfer) ---
echo "Mounting model blob..."
MOUNT_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "${AUTH[0]}" \
  "https://${REGISTRY}/v2/${TARGET_REPO}/blobs/uploads/?mount=${MODEL_LAYER_DIGEST}&from=${MODEL_REPO}")

case "$MOUNT_HTTP" in
  201) echo "  Mounted (server-side copy)" ;;
  *)   # Check if blob is already present in target repo
       HEAD_HTTP=$(curl -s -o /dev/null -w "%{http_code}" --head \
         -H "${AUTH[0]}" \
         "https://${REGISTRY}/v2/${TARGET_REPO}/blobs/${MODEL_LAYER_DIGEST}")
       if [ "$HEAD_HTTP" = "200" ]; then
         echo "  Already present in target repo"
       else
         echo "ERROR: mount failed (HTTP ${MOUNT_HTTP}), blob not in target (HTTP ${HEAD_HTTP})" >&2; exit 1
       fi ;;
esac

# --- 4. Build and push new image config ---
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "$BASE_CONFIG" | jq -c \
  --arg d "$MODEL_DIFF_ID" \
  '.rootfs.diff_ids += [$d] | .history += [{"comment":"model weights"}]' \
  > "$TMPDIR/config.json"

CONFIG_DIGEST="sha256:$(sha256_file "$TMPDIR/config.json")"
CONFIG_SIZE=$(wc -c < "$TMPDIR/config.json" | tr -d ' ')

echo "Pushing config (${CONFIG_SIZE} bytes)..."
HEADER_FILE="$TMPDIR/headers.txt"
POST_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -D "$HEADER_FILE" -X POST \
  -H "${AUTH[0]}" \
  "https://${REGISTRY}/v2/${TARGET_REPO}/blobs/uploads/")

if [ "$POST_HTTP" != "202" ]; then
  echo "ERROR: upload init failed (HTTP ${POST_HTTP})" >&2; exit 1
fi

UPLOAD_URL=$(grep -i '^location:' "$HEADER_FILE" | head -1 | tr -d '\r' | sed 's/^[Ll]ocation:[[:space:]]*//')
# Handle relative URLs
if [[ "$UPLOAD_URL" == /* ]]; then
  UPLOAD_URL="https://${REGISTRY}${UPLOAD_URL}"
fi
if [ -z "$UPLOAD_URL" ]; then
  echo "ERROR: no Location header in upload response" >&2; exit 1
fi

SEP=$([[ "$UPLOAD_URL" == *"?"* ]] && echo "&" || echo "?")
CONFIG_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
  -H "${AUTH[0]}" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@${TMPDIR}/config.json" \
  "${UPLOAD_URL}${SEP}digest=${CONFIG_DIGEST}")

if [ "$CONFIG_HTTP" != "201" ]; then
  echo "ERROR: config push failed (HTTP ${CONFIG_HTTP})" >&2; exit 1
fi
echo "  Config: ${CONFIG_DIGEST}"

# --- 5. Build and push manifest ---
echo "Pushing manifest..."
MANIFEST_MEDIA=$(echo "$BASE_MANIFEST" | jq -r '.mediaType')

NEW_MANIFEST=$(echo "$BASE_MANIFEST" | jq -c \
  --arg cd "$CONFIG_DIGEST" --argjson cs "$CONFIG_SIZE" \
  --arg ld "$MODEL_LAYER_DIGEST" --argjson ls "$MODEL_LAYER_SIZE" --arg lm "$MODEL_LAYER_MEDIA" \
  '.config.digest = $cd | .config.size = $cs |
   .layers += [{"mediaType": $lm, "digest": $ld, "size": $ls}]')

MANIFEST_HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
  -H "${AUTH[0]}" \
  -H "Content-Type: ${MANIFEST_MEDIA}" \
  -d "$NEW_MANIFEST" \
  "https://${REGISTRY}/v2/${TARGET_REPO}/manifests/${TARGET_TAG}")

if [ "$MANIFEST_HTTP" != "201" ]; then
  echo "ERROR: manifest push failed (HTTP ${MANIFEST_HTTP})" >&2; exit 1
fi
echo "Pushed ${COMBO_TAG}"

# --- 6. Verify ---
COMBO_LAST=$(crane manifest --platform "$PLATFORM" "$COMBO_TAG" | jq -r '.layers[-1].digest')
if [ "$COMBO_LAST" != "$MODEL_LAYER_DIGEST" ]; then
  echo "ERROR: digest mismatch combo=${COMBO_LAST} model=${MODEL_LAYER_DIGEST}" >&2; exit 1
fi
echo "Verified: layer digest matches"
