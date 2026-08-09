#!/usr/bin/env bash
# Start MFH Studio and get out of the way.
#
# Runs on every attach, so it has to be idempotent: reconnecting to a codespace
# that is already serving must not start a second one.
set -uo pipefail

PORT="${MFHSTUDIO_PORT:-8765}"
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG=/tmp/mfhstudio.log

alive() {
  # A 200 on the state endpoint is the only proof that matters; a listening
  # socket could be some other process.
  curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/api/state" 2>/dev/null
}

if alive; then
  echo "▸ MFH Studio is already running on port ${PORT}."
else
  echo "▸ starting MFH Studio on port ${PORT}…"
  # `--no-browser`: the browser is opened by the port forwarding rule in
  # devcontainer.json, on the *local* machine, which the container cannot do.
  ( cd "${ROOT}/tools/mfhstudio" \
    && nohup python3 -m mfhstudio --no-browser --port "${PORT}" >"${LOG}" 2>&1 & )

  # The HTTP server answers immediately; the Julia sidecar takes about ten
  # seconds more and the interface says so itself, so there is no reason to
  # wait for it here.
  for _ in $(seq 1 60); do
    alive && break
    sleep 0.5
  done
fi

if alive; then
  cat <<EOF

  ┌──────────────────────────────────────────────────────────────┐
  │  MFH Studio is running.                                      │
  │                                                              │
  │  A browser tab should have opened by itself. If it did not,  │
  │  open the PORTS tab next to the terminal and click the       │
  │  globe icon on port ${PORT}.                                    │
  │                                                              │
  │  Start with  tools/mfhstudio/examples/01_porous_schemes.jl   │
  │  via the Open… button.                                       │
  │                                                              │
  │  The badge top-right turns green once Julia has loaded       │
  │  (~10 s). Until then the 3-D view and Run are asleep.        │
  └──────────────────────────────────────────────────────────────┘

EOF
else
  echo "▸ MFH Studio did not come up. The log says:"
  tail -n 20 "${LOG}" 2>/dev/null || echo "  (no log at ${LOG})"
  echo "▸ You can start it by hand with:"
  echo "    cd tools/mfhstudio && python3 -m mfhstudio --no-browser --port ${PORT}"
fi
