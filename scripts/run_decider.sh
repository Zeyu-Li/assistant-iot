#!/usr/bin/env bash
# Run the Laya decision model behind the Jev-compatible /v1/systemone API.
#
# laya-serve is configured only through environment variables and defaults to
# 0.0.0.0 with no auth, so this script pins it to loopback. It runs in its own
# uv-managed environment, separate from the assistant's dependencies.
#
#   LAYA_DEVICE   mps (Mac), cuda (Jetson / NVIDIA), cpu      default: auto
#   PORT          must match decision.base_url in the profile  default: 8090
set -euo pipefail

LAYA_VERSION="0.5.4"

export LAYA_HOST="127.0.0.1"
export LAYA_PORT="${PORT:-8090}"
export LAYA_MODELS="english"
export LAYA_DEFAULT_MODEL="english"
export LAYA_MAX_LOADED="1"
export LAYA_PRELOAD="1"
export USE_TF="0"   # the model card notes imports can hang when TensorFlow is present

exec uvx --python 3.11 --from "laya[serve]==${LAYA_VERSION}" laya-serve
