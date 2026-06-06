#!/bin/bash
#
# Manual startup guide for RustInfer in separated processes
#
# This script shows commands for manually starting the three processes
# in separate terminals. Use this when you want fine-grained control.
#
# Usage: source this script or copy the commands to separate terminals

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}RustInfer - Manual Process Startup Guide${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""

# Configuration
MODEL="${1:-.}"  # Use current dir as model if not specified
DEVICE="${2:-cuda:0}"
LOG_LEVEL="${3:-info}"
PORT="${4:-8000}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo "  API Port: $PORT"
echo ""

# Generate unique IPC endpoints based on PID
# Use a fixed PID for manual startup (e.g., your user ID or a fixed number)
PID="${USER_PID:-12345}"
FRONTEND_EP="ipc:///tmp/rustinfer-${PID}-frontend.ipc"
WORKER_IN_EP="ipc:///tmp/rustinfer-${PID}-worker-in.ipc"
WORKER_OUT_EP="ipc:///tmp/rustinfer-${PID}-worker-out.ipc"
WORKER_CONTROL_EP="ipc:///tmp/rustinfer-${PID}-worker-control.ipc"

echo "IPC Endpoints:"
echo "  Frontend: $FRONTEND_EP"
echo "  Worker In: $WORKER_IN_EP"
echo "  Worker Out: $WORKER_OUT_EP"
echo "  Worker Control: $WORKER_CONTROL_EP"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Terminal 1: Scheduler + Worker
# ═══════════════════════════════════════════════════════════════════════════
cat << 'EOF'
═══════════════════════════════════════════════════════════════════════════
STEP 1: Start Scheduler + Worker (run in Terminal 1)
═══════════════════════════════════════════════════════════════════════════

EOF

echo -e "${CYAN}export RUST_LOG=$LOG_LEVEL${NC}"
echo -e "${CYAN}rustinfer-server \\"
echo "    --model '$MODEL' \\"
echo "    --device '$DEVICE'${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Terminal 2: API Server
# ═══════════════════════════════════════════════════════════════════════════
cat << 'EOF'
═══════════════════════════════════════════════════════════════════════════
STEP 2: Start API Server (run in Terminal 2, after Step 1 is ready)
═══════════════════════════════════════════════════════════════════════════

EOF

echo -e "${CYAN}export RUST_LOG=$LOG_LEVEL${NC}"
echo -e "${CYAN}rustinfer-api \\"
echo "    --model '$MODEL' \\"
echo "    --frontend-endpoint '$FRONTEND_EP' \\"
echo "    --port $PORT${NC}"
echo ""

echo -e "${GREEN}After both processes are running, test with:${NC}"
echo ""
echo -e "${CYAN}curl http://localhost:$PORT/v1/models${NC}"
echo ""
echo -e "${GREEN}To stop all processes:${NC}"
echo "  1. Ctrl+C in API server terminal (Terminal 2)"
echo "  2. Ctrl+C in Scheduler/Worker terminal (Terminal 1)"
echo ""
echo -e "${GREEN}Clean up IPC files:${NC}"
echo -e "${CYAN}rm -f /tmp/rustinfer-${PID}*.ipc${NC}"
echo ""
