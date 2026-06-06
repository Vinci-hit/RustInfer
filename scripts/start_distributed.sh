#!/bin/bash
#
# Start RustInfer in distributed mode: separate Scheduler/Worker and API processes
#
# Usage:
#   ./start_distributed.sh [OPTIONS]
#
# Options:
#   --model PATH              Model path (required)
#   --port PORT               API port (default: 8000)
#   --host HOST               API host (default: 0.0.0.0)
#   --device DEVICE           GPU device (default: cuda:0)
#   --log-level LEVEL         Log level (default: info)
#   --skip-api                Skip starting API server (only start scheduler+worker)
#   --help                    Show this help

set -e

# Default values
MODEL=""
PORT="8000"
HOST="0.0.0.0"
DEVICE="cuda:0"
LOG_LEVEL="info"
SKIP_API=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --skip-api)
            SKIP_API=1
            shift
            ;;
        --help)
            grep '^#' "$0" | tail -n +2 | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: --model is required${NC}"
    exit 1
fi

if [ ! -d "$MODEL" ]; then
    echo -e "${RED}Error: Model directory does not exist: $MODEL${NC}"
    exit 1
fi

# Check if binaries exist
if ! command -v rustinfer-server &> /dev/null; then
    echo -e "${YELLOW}Warning: rustinfer-server not found in PATH${NC}"
    echo "Building from source..."
    cd "$REPO_ROOT"
    cargo build --release --bin rustinfer-server
fi

if [ $SKIP_API -eq 0 ] && ! command -v rustinfer-api &> /dev/null; then
    echo -e "${YELLOW}Warning: rustinfer-api not found in PATH${NC}"
    echo "Building from source..."
    cd "$REPO_ROOT"
    cargo build --release --bin rustinfer-api
fi

if ! command -v rustinfer-scheduler &> /dev/null; then
    echo -e "${YELLOW}Warning: rustinfer-scheduler not found in PATH${NC}"
fi

if ! command -v rustinfer-worker &> /dev/null; then
    echo -e "${YELLOW}Warning: rustinfer-worker not found in PATH${NC}"
fi

# Generate unique IPC endpoints based on PID
# This allows running multiple instances in parallel
MAIN_PID=$$
FRONTEND_EP="ipc:///tmp/rustinfer-${MAIN_PID}-frontend.ipc"
WORKER_IN_EP="ipc:///tmp/rustinfer-${MAIN_PID}-worker-in.ipc"
WORKER_OUT_EP="ipc:///tmp/rustinfer-${MAIN_PID}-worker-out.ipc"
WORKER_CONTROL_EP="ipc:///tmp/rustinfer-${MAIN_PID}-worker-control.ipc"

# Create log directory
LOG_DIR="/tmp/rustinfer-${MAIN_PID}"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    RustInfer Distributed Mode - Launcher         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo "  API Port: $PORT"
echo "  Logs: $LOG_DIR"
echo ""

# Cleanup function for all spawned processes
cleanup() {
    echo -e "${YELLOW}Shutting down all processes...${NC}"
    
    # Kill scheduler and worker (they were started as child processes)
    if [ ! -z "$SCHEDULER_PID" ]; then
        echo "Stopping scheduler (PID: $SCHEDULER_PID)..."
        kill -TERM $SCHEDULER_PID 2>/dev/null || true
        sleep 1
        kill -9 $SCHEDULER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$WORKER_PID" ]; then
        echo "Stopping worker (PID: $WORKER_PID)..."
        kill -TERM $WORKER_PID 2>/dev/null || true
        sleep 1
        kill -9 $WORKER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$API_PID" ]; then
        echo "Stopping API server (PID: $API_PID)..."
        kill -TERM $API_PID 2>/dev/null || true
        sleep 1
        kill -9 $API_PID 2>/dev/null || true
    fi
    
    # Clean up IPC files
    rm -f /tmp/rustinfer-${MAIN_PID}*.ipc
    
    echo -e "${GREEN}All processes stopped.${NC}"
}

# Set up trap for cleanup on exit
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Start Scheduler + Worker via rustinfer-server
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${GREEN}[1/3] Starting Scheduler & Worker...${NC}"

RUST_LOG=$LOG_LEVEL rustinfer-server \
    --model "$MODEL" \
    --device "$DEVICE" \
    > "$LOG_DIR/scheduler-worker.log" 2>&1 &

SCHEDULER_WORKER_PID=$!
SCHEDULER_PID=$SCHEDULER_WORKER_PID
WORKER_PID=$SCHEDULER_WORKER_PID

echo -e "${GREEN}  Scheduler & Worker started (PID: $SCHEDULER_WORKER_PID)${NC}"
echo "  Log: $LOG_DIR/scheduler-worker.log"

# Wait for IPC sockets to be created (indicates scheduler is ready)
echo -e "${YELLOW}  Waiting for scheduler to initialize...${NC}"
for i in {1..30}; do
    if [ -S "${FRONTEND_EP#ipc://}" ]; then
        echo -e "${GREEN}  Scheduler ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}  Timeout waiting for scheduler to initialize${NC}"
        exit 1
    fi
    sleep 1
done

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Start API Server (optional)
# ═══════════════════════════════════════════════════════════════════════════
if [ $SKIP_API -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[2/3] Starting API Server...${NC}"
    
    sleep 1  # Give scheduler extra time
    
    RUST_LOG=$LOG_LEVEL rustinfer-api \
        --model "$MODEL" \
        --frontend-endpoint "$FRONTEND_EP" \
        --host "$HOST" \
        --port "$PORT" \
        > "$LOG_DIR/api.log" 2>&1 &
    
    API_PID=$!
    echo -e "${GREEN}  API Server started (PID: $API_PID)${NC}"
    echo "  Log: $LOG_DIR/api.log"
    echo -e "${GREEN}  API listening on http://$HOST:$PORT${NC}"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Monitor all processes
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}[3/3] All services running. Press Ctrl+C to stop.${NC}"
echo ""
echo "Process IDs:"
echo "  Scheduler & Worker: $SCHEDULER_WORKER_PID"
if [ $SKIP_API -eq 0 ]; then
    echo "  API Server: $API_PID"
fi
echo ""
echo "Logs:"
echo "  tail -f $LOG_DIR/scheduler-worker.log"
if [ $SKIP_API -eq 0 ]; then
    echo "  tail -f $LOG_DIR/api.log"
fi
echo ""

# Monitor processes
while true; do
    # Check scheduler+worker
    if ! kill -0 $SCHEDULER_WORKER_PID 2>/dev/null; then
        echo -e "${RED}Scheduler/Worker process died!${NC}"
        exit 1
    fi
    
    # Check API server if enabled
    if [ $SKIP_API -eq 0 ]; then
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "${RED}API Server process died!${NC}"
            exit 1
        fi
    fi
    
    sleep 1
done
