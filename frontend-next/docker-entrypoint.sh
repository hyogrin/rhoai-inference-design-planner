#!/bin/sh
set -e

# ── Replace build-time placeholders with runtime env vars ─────────
# Next.js inlines NEXT_PUBLIC_* at build time. We build with placeholder
# strings and swap them here so one image works on any cluster.

PLACEHOLDER_API="__NEXT_PUBLIC_API_URL_PLACEHOLDER__"
PLACEHOLDER_AGENT="__NEXT_PUBLIC_AGENT_URL_PLACEHOLDER__"

API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:7001}"
AGENT_URL="${NEXT_PUBLIC_AGENT_URL:-${API_URL}/agent}"

echo "▸ Injecting runtime API URLs:"
echo "  NEXT_PUBLIC_API_URL   = ${API_URL}"
echo "  NEXT_PUBLIC_AGENT_URL = ${AGENT_URL}"

find /app/.next -name '*.js' -exec sed -i \
  -e "s|${PLACEHOLDER_API}|${API_URL}|g" \
  -e "s|${PLACEHOLDER_AGENT}|${AGENT_URL}|g" \
  {} +

echo "✔ API URLs injected"

exec node server.js
