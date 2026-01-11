#!/bin/bash
# RustInfer 分离式架构启动脚本

set -e

# 配置
MODEL_PATH="${MODEL_PATH:-/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct}"
ZMQ_ENDPOINT="${ZMQ_ENDPOINT:-ipc:///tmp/rustinfer.ipc}"
API_PORT="${API_PORT:-8000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

echo "========================================"
echo "  RustInfer 分离式架构启动"
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "ZMQ地址: $ZMQ_ENDPOINT"
echo "API端口: $API_PORT"
echo "Batch大小: $BATCH_SIZE"
echo ""

# 清理旧的IPC文件
if [[ $ZMQ_ENDPOINT == ipc://* ]]; then
    IPC_FILE=$(echo $ZMQ_ENDPOINT | sed 's|ipc://||')
    if [ -f "$IPC_FILE" ]; then
        echo "清理旧的IPC文件: $IPC_FILE"
        rm -f "$IPC_FILE"
    fi
fi

# 构建项目
echo "[1/3] 构建项目..."
cargo build --release

# 启动Engine进程 (后台)
echo "[2/3] 启动Engine进程..."
./target/release/rustinfer-engine \
    --model "$MODEL_PATH" \
    --device cuda:0 \
    --zmq-endpoint "$ZMQ_ENDPOINT" \
    --batch-size $BATCH_SIZE \
    --max-queue-size 128 \
    &

ENGINE_PID=$!
echo "Engine进程PID: $ENGINE_PID"

# 等待Engine启动
echo "等待Engine启动..."
sleep 5

# 启动API Server
echo "[3/3] 启动API Server..."
./target/release/rustinfer-server \
    --port $API_PORT \
    --engine-endpoint "$ZMQ_ENDPOINT" \
    &

SERVER_PID=$!
echo "API Server进程PID: $SERVER_PID"

echo ""
echo "✅ 启动完成!"
echo "   Engine PID: $ENGINE_PID"
echo "   Server PID: $SERVER_PID"
echo ""
echo "测试命令:"
echo "  curl http://localhost:$API_PORT/health"
echo "  curl http://localhost:$API_PORT/ready"
echo ""
echo "停止命令:"
echo "  kill $ENGINE_PID $SERVER_PID"
echo ""

# 等待进程
wait
