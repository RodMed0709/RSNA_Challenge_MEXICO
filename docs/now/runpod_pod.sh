#!/usr/bin/env bash
set -euo pipefail

API_BASE="https://rest.runpod.io/v1/pods"
GRAPHQL_URL="https://api.runpod.io/graphql"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
NETWORK_VOLUME_ID="lu0c3buems"
DATA_CENTER_ID="EU-RO-1"

usage() {
  cat <<'EOF'
Usage:
  runpod_pod.sh list
  runpod_pod.sh balance
  runpod_pod.sh create <name> <gpuType>
  runpod_pod.sh ssh-info <podId>
  runpod_pod.sh stop <podId>
  runpod_pod.sh delete <podId>

RUNPOD_API_KEY must already be exported. create reads the SSH public key
from RUNPOD_SSH_PUBLIC_KEY, ~/.ssh/id_rsa.pub, or ~/.ssh/id_ed25519.pub.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

[[ -n "${RUNPOD_API_KEY:-}" ]] || die "RUNPOD_API_KEY is not set"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  die "python3 or python is required to parse JSON"
fi

api() {
  local method=$1 url=$2 data=${3:-}
  local args=(--fail --silent --show-error --request "$method" --url "$url"
    --header "Authorization: Bearer ${RUNPOD_API_KEY}")
  if [[ -n "$data" ]]; then
    args+=(--header "Content-Type: application/json" --data "$data")
  fi
  curl "${args[@]}"
}

require_pod_id() {
  local pod_id=${1:-}
  [[ -n "$pod_id" ]] || die "podId is required"
  [[ "$pod_id" =~ ^[A-Za-z0-9_-]+$ ]] || die "invalid podId: $pod_id"
}

command=${1:-}
case "$command" in
  list)
    api GET "$API_BASE" | "$PYTHON" -c 'import json, sys
pods = json.load(sys.stdin)
pods = pods if isinstance(pods, list) else pods.get("pods", pods.get("data", []))
for pod in pods:
    gpu = pod.get("gpu") or {}
    machine = pod.get("machine") or {}
    gpu_type = machine.get("gpuType") or {}
    gpu_name = (gpu.get("displayName") if isinstance(gpu, dict) else gpu) or machine.get("gpuDisplayName") or gpu_type.get("displayName") or "-"
    print("\t".join(str(value) for value in (pod.get("id", "-"), pod.get("name", "-"), pod.get("desiredStatus", "-"), pod.get("costPerHr", "-"), gpu_name)))'
    ;;
  balance)
    payload='{"query":"{myself{clientBalance currentSpendPerHr}}"}'
    api POST "$GRAPHQL_URL" "$payload" | "$PYTHON" -c 'import json, sys
data = json.load(sys.stdin).get("data", {}).get("myself", {})
print(f"clientBalance\t{data.get('"'"'clientBalance'"'"', '"'"'-'"'"')}")
print(f"currentSpendPerHr\t{data.get('"'"'currentSpendPerHr'"'"', '"'"'-'"'"')}")'
    ;;
  create)
    name=${2:-}
    gpu_type=${3:-}
    [[ -n "$name" ]] || die "name is required"
    [[ -n "$gpu_type" ]] || die "gpuType is required"

    public_key=${RUNPOD_SSH_PUBLIC_KEY:-}
    if [[ -z "$public_key" && -s "$HOME/.ssh/id_rsa.pub" ]]; then
      public_key=$(<"$HOME/.ssh/id_rsa.pub")
    elif [[ -z "$public_key" && -s "$HOME/.ssh/id_ed25519.pub" ]]; then
      public_key=$(<"$HOME/.ssh/id_ed25519.pub")
    fi
    [[ -n "$public_key" ]] || die "no SSH public key found"

    payload=$("$PYTHON" - "$name" "$gpu_type" "$public_key" "$IMAGE" "$NETWORK_VOLUME_ID" "$DATA_CENTER_ID" <<'PY'
import json
import sys

name, gpu_type, public_key, image, volume_id, data_center = sys.argv[1:]
print(json.dumps({
    "name": name,
    "imageName": image,
    "gpuTypeIds": [gpu_type],
    "gpuCount": 1,
    "cloudType": "SECURE",
    "networkVolumeId": volume_id,
    "dataCenterIds": [data_center],
    "containerDiskInGb": 120,
    "ports": ["22/tcp"],
    "volumeMountPath": "/workspace",
    "supportPublicIp": True,
    "env": {"PUBLIC_KEY": public_key},
}))
PY
)
    api POST "$API_BASE" "$payload" | "$PYTHON" -c 'import json, sys
pod = json.load(sys.stdin)
if not pod.get("id"):
    raise SystemExit("RunPod create failed: " + json.dumps(pod))
print("\t".join(str(value) for value in (pod["id"], pod.get("name", "-"), pod.get("desiredStatus", "-"))))'
    ;;
  ssh-info)
    pod_id=${2:-}
    require_pod_id "$pod_id"
    api GET "$API_BASE/$pod_id" | "$PYTHON" -c 'import json, sys
pod = json.load(sys.stdin)
ip = pod.get("publicIp")
port = (pod.get("portMappings") or {}).get("22")
if not ip or not port:
    raise SystemExit("SSH is not ready for this pod")
print(f"ssh -p {port} root@{ip}")'
    ;;
  stop)
    pod_id=${2:-}
    require_pod_id "$pod_id"
    api POST "$API_BASE/$pod_id/stop" >/dev/null
    printf 'stopped\t%s\n' "$pod_id"
    ;;
  delete)
    pod_id=${2:-}
    require_pod_id "$pod_id"
    api DELETE "$API_BASE/$pod_id"
    printf 'deleted\t%s\n' "$pod_id"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
