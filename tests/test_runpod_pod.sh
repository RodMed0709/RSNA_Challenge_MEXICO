#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT="$ROOT/docs/now/runpod_pod.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack=$1 needle=$2
  [[ "$haystack" == *"$needle"* ]] || fail "expected '$needle' in: $haystack"
}

curl() {
  local args=" $* "
  case "$args" in
    *" https://rest.runpod.io/v1/pods "*)
      if [[ "$args" == *" --request POST "* ]]; then
        [[ "$args" == *'"name": "test-pod"'* ]] || return 90
        [[ "$args" == *'"gpuTypeIds": ["NVIDIA RTX 6000 Ada Generation"]'* ]] || return 91
        [[ "$args" == *'"networkVolumeId": "lu0c3buems"'* ]] || return 92
        [[ "$args" == *'"dataCenterIds": ["EU-RO-1"]'* ]] || return 93
        [[ "$args" == *'"ports": ["22/tcp"]'* ]] || return 94
        [[ "$args" == *'"PUBLIC_KEY": "ssh-ed25519 TEST"'* ]] || return 95
        printf '%s\n' '{"id":"new-pod","name":"test-pod","desiredStatus":"RUNNING"}'
      else
        printf '%s\n' '[{"id":"pod-1","name":"gpu-pod","desiredStatus":"RUNNING","costPerHr":0.72,"gpu":{"displayName":"NVIDIA RTX 6000 Ada Generation"}},{"id":"pod-2","name":"cpu-pod","desiredStatus":"EXITED","costPerHr":0.092,"gpu":null}]'
      fi
      ;;
    *" https://api.runpod.io/graphql "*)
      [[ "$args" == *'myself{clientBalance currentSpendPerHr}'* ]] || return 96
      printf '%s\n' '{"data":{"myself":{"clientBalance":35.5,"currentSpendPerHr":0.165}}}'
      ;;
    *" https://rest.runpod.io/v1/pods/pod-1 "*)
      if [[ "$args" == *" --request DELETE "* ]]; then
        return 0
      fi
      printf '%s\n' '{"id":"pod-1","name":"gpu-pod","desiredStatus":"RUNNING","publicIp":"203.0.113.7","portMappings":{"22":22022}}'
      ;;
    *" https://rest.runpod.io/v1/pods/pod-1/stop "*)
      [[ "$args" == *" --request POST "* ]] || return 97
      ;;
    *)
      echo "unexpected curl call:$args" >&2
      return 98
      ;;
  esac
}
export -f curl
export RUNPOD_API_KEY=test-key
export RUNPOD_SSH_PUBLIC_KEY='ssh-ed25519 TEST'

list_output=$(bash "$SCRIPT" list)
assert_contains "$list_output" $'pod-1\tgpu-pod\tRUNNING\t0.72\tNVIDIA RTX 6000 Ada Generation'
assert_contains "$list_output" $'pod-2\tcpu-pod\tEXITED\t0.092\t-'

balance_output=$(bash "$SCRIPT" balance)
assert_contains "$balance_output" $'clientBalance\t35.5'
assert_contains "$balance_output" $'currentSpendPerHr\t0.165'

ssh_output=$(bash "$SCRIPT" ssh-info pod-1)
[[ "$ssh_output" == 'ssh -p 22022 root@203.0.113.7' ]] || fail "unexpected ssh-info: $ssh_output"

create_output=$(bash "$SCRIPT" create test-pod 'NVIDIA RTX 6000 Ada Generation')
assert_contains "$create_output" $'new-pod\ttest-pod\tRUNNING'

stop_output=$(bash "$SCRIPT" stop pod-1)
[[ "$stop_output" == $'stopped\tpod-1' ]] || fail "unexpected stop output: $stop_output"

delete_output=$(bash "$SCRIPT" delete pod-1)
[[ "$delete_output" == $'deleted\tpod-1' ]] || fail "unexpected delete output: $delete_output"

echo "PASS: runpod_pod.sh command contract"
