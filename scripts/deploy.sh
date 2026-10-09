#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# deploy.sh — Build, push, and deploy to OpenShift cluster
#
# Usage:
#   ./scripts/deploy.sh              # deploy both backend + frontend
#   ./scripts/deploy.sh backend      # deploy backend only
#   ./scripts/deploy.sh frontend     # deploy frontend only
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Registry & namespace ──────────────────────────────────────────
REGISTRY="${REGISTRY:-quay.io/rh-ee-hyochoi}"
BACKEND_IMAGE="${REGISTRY}/inference-planner-backend:latest"
FRONTEND_IMAGE="${REGISTRY}/inference-planner-frontend:latest"
NAMESPACE="${NAMESPACE:-inference-planner}"
PLATFORM="${PLATFORM:-linux/amd64}"

# ── Derive API URLs from cluster route (or use explicit override) ─
CLUSTER_DOMAIN="${CLUSTER_DOMAIN:-}"
if [[ -z "$CLUSTER_DOMAIN" ]]; then
  CLUSTER_DOMAIN=$(oc get route inference-planner-api -n "$NAMESPACE" \
    -o jsonpath='{.spec.host}' 2>/dev/null || true)
fi
if [[ -z "$CLUSTER_DOMAIN" ]]; then
  echo "ERROR: Cannot determine backend route. Set CLUSTER_DOMAIN or ensure 'oc' is logged in." >&2
  exit 1
fi

NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-https://${CLUSTER_DOMAIN}}"
NEXT_PUBLIC_AGENT_URL="${NEXT_PUBLIC_AGENT_URL:-https://${CLUSTER_DOMAIN}/agent}"

# ── What to deploy ────────────────────────────────────────────────
TARGET="${1:-all}"

info()  { echo -e "\033[36m▸ $*\033[0m"; }
ok()    { echo -e "\033[32m✔ $*\033[0m"; }
fail()  { echo -e "\033[31m✖ $*\033[0m" >&2; exit 1; }

# ── Backend ───────────────────────────────────────────────────────
build_backend() {
  info "Building backend image…"
  docker build --platform "$PLATFORM" \
    -t "$BACKEND_IMAGE" \
    -f "$PROJECT_ROOT/Dockerfile.backend" \
    "$PROJECT_ROOT"
  ok "Backend image built"
}

push_backend() {
  info "Pushing backend image…"
  docker push "$BACKEND_IMAGE"
  ok "Backend pushed: $BACKEND_IMAGE"
}

deploy_backend() {
  info "Rolling out backend…"
  oc rollout restart deployment/backend -n "$NAMESPACE"
  oc rollout status  deployment/backend -n "$NAMESPACE" --timeout=120s
  ok "Backend deployed"
}

# ── Frontend ──────────────────────────────────────────────────────
build_frontend() {
  info "Building frontend image…"
  info "  NEXT_PUBLIC_API_URL   = $NEXT_PUBLIC_API_URL"
  info "  NEXT_PUBLIC_AGENT_URL = $NEXT_PUBLIC_AGENT_URL"
  docker build --platform "$PLATFORM" \
    --build-arg "NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL" \
    --build-arg "NEXT_PUBLIC_AGENT_URL=$NEXT_PUBLIC_AGENT_URL" \
    -t "$FRONTEND_IMAGE" \
    -f "$PROJECT_ROOT/frontend-next/Dockerfile" \
    "$PROJECT_ROOT/frontend-next/"
  ok "Frontend image built (API URLs baked in)"
}

push_frontend() {
  info "Pushing frontend image…"
  docker push "$FRONTEND_IMAGE"
  ok "Frontend pushed: $FRONTEND_IMAGE"
}

deploy_frontend() {
  info "Rolling out frontend…"
  oc rollout restart deployment/frontend -n "$NAMESPACE"
  oc rollout status  deployment/frontend -n "$NAMESPACE" --timeout=120s
  ok "Frontend deployed"
}

# ── Orchestrate ───────────────────────────────────────────────────
case "$TARGET" in
  backend)
    build_backend && push_backend && deploy_backend
    ;;
  frontend)
    build_frontend && push_frontend && deploy_frontend
    ;;
  all)
    build_backend  && push_backend  && deploy_backend
    build_frontend && push_frontend && deploy_frontend
    ;;
  *)
    echo "Usage: $0 [all|backend|frontend]" >&2
    exit 1
    ;;
esac

echo ""
ok "Deploy complete ($TARGET)"
