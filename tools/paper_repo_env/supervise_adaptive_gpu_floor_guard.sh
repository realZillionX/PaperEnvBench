#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/gpu_occupancy_guard}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
mkdir -p "${RUN_DIR}"

SUPERVISOR_PID_FILE="${SUPERVISOR_PID_FILE:-${RUN_DIR}/adaptive_floor_supervisor.pid}"
CHILD_PID_FILE="${CHILD_PID_FILE:-${RUN_DIR}/adaptive_floor_guard.pid}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${RUN_DIR}/adaptive_floor_supervisor.log}"

stop_requested=0
child_pid=""

log_line() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"${SUPERVISOR_LOG}"
}

cleanup() {
  stop_requested=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  rm -f "${SUPERVISOR_PID_FILE}" "${CHILD_PID_FILE}"
}

trap cleanup INT TERM EXIT

echo "$$" >"${SUPERVISOR_PID_FILE}"
log_line "supervisor_start pid=$$"

while [[ "${stop_requested}" -eq 0 ]]; do
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/paper_repo_env/adaptive_gpu_floor_guard.py" "$@" &
  child_pid="$!"
  echo "${child_pid}" >"${CHILD_PID_FILE}"
  log_line "child_start pid=${child_pid}"

  set +e
  wait "${child_pid}"
  status="$?"
  set -e

  if [[ "${stop_requested}" -ne 0 ]]; then
    break
  fi
  log_line "child_exit pid=${child_pid} status=${status}; restarting_after=5s"
  sleep 5
done
