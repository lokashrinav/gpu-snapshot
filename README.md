# GPU Snapshot

Multi-GPU checkpoint/restore for gVisor containers. Verified on 2x H100 with NCCL.

## How It Works

The app calls NVIDIA's cuda-checkpoint API on itself via signal handlers:

1. **SIGUSR1** → `cuCheckpointProcessLock` + `cuCheckpointProcessCheckpoint`
   - Atomically freezes all GPUs (prevents NCCL deadlocks)
   - Copies GPU VRAM to host memory (~8x allocation size)
   - Terminates CUDA session

2. **`runsc checkpoint`** → serializes process memory (which now contains GPU state) to disk

3. **`runsc restore`** → loads process memory back (new sentry, fresh driver session)

4. **SIGUSR2** → `cuCheckpointProcessRestore` + `cuCheckpointProcessUnlock`
   - Reads GPU state from host memory
   - Creates new CUDA session
   - App continues — GPU memory intact

## Files

| File | Purpose |
|------|---------|
| `gpu_snapshot.c` | The complete solution — signal handlers + example app |
| `signal_helper.go` | gVisor save-restore-exec helper (sends signals to the app) |

That's it. Two files.

## Build

```bash
# App (or link gpu_snapshot.c into your own app)
gcc gpu_snapshot.c -o gpu_snapshot -ldl

# Helper binary (goes inside the container)
go build -o signal_helper signal_helper.go
```

## Usage

```bash
# Enable persistence mode
nvidia-smi -pm 1

# Start container
runsc --nvproxy --nvproxy-driver-version=580.105.08 \
    run --bundle /path/to/bundle $CONTAINER_ID

# Checkpoint (GPU state → host memory → disk, sentry exits)
runsc checkpoint \
    --save-restore-exec-argv=/bin/signal_helper \
    --save-restore-exec-timeout=120s \
    --image-path=/tmp/checkpoint \
    $CONTAINER_ID

# Cold restore (new sentry, GPU state from host memory)
runsc --nvproxy --nvproxy-driver-version=580.105.08 \
    restore \
    --image-path=/tmp/checkpoint \
    --bundle=/path/to/bundle \
    $RESTORED_CONTAINER_ID
```

## Integrating Into Your App

Add these three things to your application:

```c
#include "gpu_snapshot.h"  // or inline the functions

// 1. At startup:
install_checkpoint_handlers();

// 2. In your main loop, skip GPU work while locked:
if (is_gpu_locked()) continue;

// 3. That's it. gVisor handles the rest.
```

For Python/PyTorch, call the cuda-checkpoint functions via ctypes:

```python
import signal, ctypes, os

libcuda = ctypes.CDLL("libcuda.so.1")

class Args(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char * 64)]

def checkpoint_handler(signum, frame):
    args = Args()
    libcuda.cuCheckpointProcessLock(os.getpid(), ctypes.byref(args))
    args = Args()
    libcuda.cuCheckpointProcessCheckpoint(os.getpid(), ctypes.byref(args))
    with open("/tmp/.gpu_ckpt_done", "w") as f: f.write("0\n")

def restore_handler(signum, frame):
    args = Args()
    libcuda.cuCheckpointProcessRestore(os.getpid(), ctypes.byref(args))
    args = Args()
    libcuda.cuCheckpointProcessUnlock(os.getpid(), ctypes.byref(args))
    with open("/tmp/.gpu_ckpt_done", "w") as f: f.write("0\n")

signal.signal(signal.SIGUSR1, checkpoint_handler)
signal.signal(signal.SIGUSR2, restore_handler)
```

## Test Results

All verified on 2x H100 SXM5:

| Test | Sentry | Result |
|------|--------|--------|
| Single-GPU cold restore | NEW | restore=0, 0xBEEF1234 ✓ |
| Multi-GPU cold restore | NEW | restore=0, 0xCAFE0000 + 0xCAFE0001 ✓ |
| Multi-GPU + NCCL cold restore | NEW | restore=0, patterns + allreduce ✓ |
| PyTorch model | Same | restore=0, tensors + model ✓ |

## Requirements

- NVIDIA driver 570+ (cuda-checkpoint API)
- gVisor with nvproxy
- Turing+ GPU (T4, A100, H100 — not V100)
- `nvidia-smi -pm 1` (persistence mode)

## How Modal Does It

This is the same approach Modal uses for their GPU Memory Snapshots. From their [blog](https://modal.com/blog/gpu-mem-snapshots):

> The driver checkpoints device memory in host memory so that it can be checkpointed to disk by a host-side checkpointing system. Then, once the host-side system has restored the host memory, including the device checkpoint, the driver restores the device memory.
