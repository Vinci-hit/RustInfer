#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Both commands use the project's pinned dependencies.
npm run build:css
npm run tailwind &
TAILWIND_PID=$!
trap 'kill "$TAILWIND_PID" 2>/dev/null || true' EXIT INT TERM

dx serve --platform web --port 3000 "$@"
