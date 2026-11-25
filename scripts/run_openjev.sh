#!/usr/bin/env bash
# Run OpenJev-1.7B (Apache-2.0) behind the Jev-compatible /v1/systemone API using its
# reference server, wev-ai, in an isolated uv-managed environment (needs Python 3.11+).
#
# llama.cpp serves /v1/systemone natively, but only for GGUFs converted with the
# decision-head metadata; the community OpenJev-1.7B GGUF lacks it ("not a decision model").
#
#   DEVICE   mps (Mac), cuda (Jetson / NVIDIA), cpu      default: best available
#   DTYPE    fp16 | bf16 | fp32                          default: fp16 on mps/cuda
#   PORT     must match decision.base_url in the profile  default: 8092
set -euo pipefail

WEV_VERSION="0.2.2"
MODEL="alanhuangya/OpenJev-1.7B"

args=(serve --model "$MODEL" --host 127.0.0.1 --port "${PORT:-8092}" --dtype "${DTYPE:-fp16}")
[[ -n "${DEVICE:-}" ]] && args+=(--device "$DEVICE")

exec uvx --python 3.11 --from "wev-ai[serve]==${WEV_VERSION}" wev "${args[@]}"
