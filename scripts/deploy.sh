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

info()  { echo -e "\033[36m▸ $*\033[0m"; }
ok()    { echo -e "\033[32m✔ $*\033[0m"; }
fail()  { echo -e "\033[31m✖ $*\033[0m" >&2; exit 1; }

# ── Auto-detect cluster apps domain from oc login ─────────────────
# Reads the ingress domain from the logged-in cluster so you only need
# to run: oc login … && ./scripts/deploy.sh
APPS_DOMAIN="${APPS_DOMAIN:-}"
if [[ -z "$APPS_DOMAIN" ]]; then
  APPS_DOMAIN=$(oc get ingresses.config/cluster \
    -o jsonpath='{.spec.domain}' 2>/dev/null || true)
fi
if [[ -z "$APPS_DOMAIN" ]]; then
  fail "Cannot detect cluster apps domain. Run 'oc login' first or set APPS_DOMAIN."
fi

ROUTE_PREFIX="${ROUTE_PREFIX:-inference-planner-api}"
NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-https://${ROUTE_PREFIX}.${APPS_DOMAIN}}"
NEXT_PUBLIC_AGENT_URL="${NEXT_PUBLIC_AGENT_URL:-${NEXT_PUBLIC_API_URL}/agent}"

info "Cluster apps domain : $APPS_DOMAIN"
info "NEXT_PUBLIC_API_URL : $NEXT_PUBLIC_API_URL"
info "NEXT_PUBLIC_AGENT_URL: $NEXT_PUBLIC_AGENT_URL"

# ── What to deploy ────────────────────────────────────────────────
TARGET="${1:-all}"

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
  info "Rolling out backend (APPS_DOMAIN=$APPS_DOMAIN)…"
  oc rollout restart deployment/backend -n "$NAMESPACE"
  oc rollout status  deployment/backend -n "$NAMESPACE" --timeout=120s
  ok "Backend deployed"
}

# ── Frontend ──────────────────────────────────────────────────────
build_frontend() {
  info "Building frontend image (cluster-agnostic)…"
  docker build --platform "$PLATFORM" \
    -t "$FRONTEND_IMAGE" \
    -f "$PROJECT_ROOT/frontend-next/Dockerfile" \
    "$PROJECT_ROOT/frontend-next/"
  ok "Frontend image built (API URLs injected at runtime via env vars)"
}

push_frontend() {
  info "Pushing frontend image…"
  docker push "$FRONTEND_IMAGE"
  ok "Frontend pushed: $FRONTEND_IMAGE"
}

deploy_frontend() {
  info "Setting frontend API URLs and rolling out…"
  oc set env deployment/frontend \
    NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" \
    NEXT_PUBLIC_AGENT_URL="$NEXT_PUBLIC_AGENT_URL" \
    -n "$NAMESPACE" 2>/dev/null || true
  oc rollout restart deployment/frontend -n "$NAMESPACE"
  oc rollout status  deployment/frontend -n "$NAMESPACE" --timeout=120s
  ok "Frontend deployed (API → $NEXT_PUBLIC_API_URL)"
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
