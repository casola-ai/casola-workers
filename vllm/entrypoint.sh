#!/bin/bash

set -e

WORKER_PID=""
SHUTDOWN_INITIATED=0

cleanup() {
    if [ $SHUTDOWN_INITIATED -eq 1 ]; then
        return
    fi
    SHUTDOWN_INITIATED=1

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Cleanup initiated"

    if [ -n "$WORKER_PID" ] && kill -0 $WORKER_PID 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Sending SIGTERM to worker (PID: $WORKER_PID)"
        kill -TERM $WORKER_PID 2>/dev/null || true

        GRACE_PERIOD=${CASOLA_SHUTDOWN_GRACE_PERIOD:-5}
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting up to ${GRACE_PERIOD}s for worker to shutdown gracefully"

        for i in $(seq 1 $GRACE_PERIOD); do
            if ! kill -0 $WORKER_PID 2>/dev/null; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker shutdown gracefully"
                break
            fi
            sleep 1
        done

        if kill -0 $WORKER_PID 2>/dev/null; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Force killing worker"
            kill -9 $WORKER_PID 2>/dev/null || true
        fi
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Cleanup complete"
}

trap cleanup SIGTERM SIGINT EXIT

# --- CUDA / GPU diagnostics ---
echo "========== CUDA/GPU Diagnostics =========="

echo "--- Host driver ---"
nvidia-smi --query-gpu=driver_version,name,memory.total,compute_cap --format=csv 2>&1 || echo "nvidia-smi FAILED"
cat /proc/driver/nvidia/version 2>/dev/null || echo "/proc/driver/nvidia/version not available"

echo "--- CUDA toolkit ---"
nvcc --version 2>/dev/null || echo "nvcc not found"

echo "--- LD_LIBRARY_PATH ---"
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

echo "--- Compat libs ---"
ls -la /usr/local/cuda/compat/ 2>/dev/null || echo "No /usr/local/cuda/compat dir"

echo "--- ldconfig CUDA/NVIDIA entries ---"
ldconfig -p 2>/dev/null | grep -iE "libcuda|libcudart|libnvidia-ml" || echo "none found"

echo "--- Actual libcuda.so resolution ---"
python3 -c "
import ctypes, ctypes.util
# Which libcuda.so the linker finds
path = ctypes.util.find_library('cuda')
print(f'ctypes.util.find_library(cuda) = {path}')

# Load NVML to read driver version
try:
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    nvml.nvmlInit_v2()
    buf = ctypes.create_string_buffer(80)
    nvml.nvmlSystemGetDriverVersion(buf, 80)
    print(f'NVML driver version: {buf.value.decode()}')
    ver = ctypes.c_uint()
    nvml.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(ver))
    major = ver.value // 1000
    minor = (ver.value % 1000) // 10
    print(f'NVML CUDA driver version: {major}.{minor}')
    nvml.nvmlShutdown()
except Exception as e:
    print(f'NVML probe failed: {e}')

# PyTorch/CUDA info
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA compiled version: {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'cuDNN: {torch.backends.cudnn.version()}')
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {p.name}, {p.total_mem/1024**3:.1f}GB, SM {p.major}.{p.minor}')
except Exception as e:
    print(f'PyTorch probe failed: {e}')
" 2>&1

echo "--- Loaded CUDA libraries (from /proc) ---"
python3 -c "
import ctypes
try:
    ctypes.CDLL('libcuda.so.1')
except: pass
with open('/proc/self/maps') as f:
    seen = set()
    for line in f:
        if any(x in line for x in ['libcuda', 'libcudart', 'libnvidia-ml', 'libnccl']):
            parts = line.split()
            path = parts[-1] if len(parts) >= 6 else ''
            if path and path not in seen:
                seen.add(path)
                print(path)
" 2>&1

echo "========== End Diagnostics =========="

# Create the unversioned libcuda.so symlink needed by Triton's -lcuda linker flag.
# The NVIDIA container runtime injects libcuda.so.1 at runtime but not the dev symlink.
if [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Created libcuda.so symlink"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting worker (worker will manage vLLM process internally)"
python3 -u /app/worker.py &
WORKER_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker started (PID: $WORKER_PID)"

# Wait for worker process to exit
wait $WORKER_PID
WORKER_EXIT_CODE=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') - Worker exited with code: $WORKER_EXIT_CODE"
exit $WORKER_EXIT_CODE
