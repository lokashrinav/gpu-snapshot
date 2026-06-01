# GPU Snapshot Research Log

Everything we tried, everything we learned, and how we arrived at the working solution. This documents the full journey across ~25 Lambda Labs instances (H100, A100, T4, V100) over multiple sessions.

## The Goal

Checkpoint a GPU container in gVisor — including multi-GPU NCCL state — kill the sentry, start a fresh sentry, restore, and have the app continue as if nothing happened. This is what Modal calls "GPU Memory Snapshots."

## The Three Domains

The problem breaks into three domains:

1. **GPU Dumping** — getting GPU VRAM out of the GPU and into host memory
2. **CPU + GPU State Snapshots** — serializing the process (which now contains GPU state) to disk and restoring it
3. **Deadlock Prevention** — coordinating multi-GPU freeze so NCCL doesn't deadlock

---

## Key Discovery: GPU State Goes to Host Memory

The single most important finding. When you call `cuCheckpointProcessCheckpoint`, the NVIDIA driver:

1. Walks every GPU memory allocation owned by the process
2. DMA transfers the contents from GPU VRAM to newly allocated anonymous pages in host memory
3. Records the GPU virtual address of each allocation
4. Saves context state, stream state, event state, loaded modules
5. Terminates the CUDA session — GPUs are released

**Evidence**: We monitored RSS before and after checkpoint. For 64MB of GPU data, RSS grew by 519MB. The GPU state is in the process's address space as regular pages.

This means gVisor doesn't need to know anything about GPU state. It just serializes process memory (which now includes GPU state) to disk using its normal checkpoint mechanism.

On restore, `cuCheckpointProcessRestore` reads from those same host memory pages, allocates new GPU memory at the ORIGINAL virtual addresses, DMA transfers the data back, and recreates the CUDA session. This works even with a new sentry and new driver session because the GPU state was in process memory, not the driver.

---

## Experiments That Failed

### Experiment 1: Cross-Process Checkpoint (Helper → App)

**Hypothesis**: A helper binary running alongside the app can call `cuCheckpointProcessCheckpoint(app_pid, args)` to checkpoint the app's GPU state.

**What we tried**:
- Built a helper binary (`gvisor-gpu-ckpt`) that targeted PID 1 (the app)
- Helper called `cuCheckpointProcessLock(1, args)` then `cuCheckpointProcessCheckpoint(1, args)`

**Result**: `cuCheckpointProcessCheckpoint` returned **401** (CUDA_ERROR_INVALID_VALUE). Every time. The lock also returned 801.

**Root cause**: The NVIDIA driver binds checkpoint data to process identity. The checkpoint data structure is tied to the calling process's internal driver state. Cross-process checkpoint is fundamentally not supported.

**Time spent**: ~20 hours across multiple approaches.

### Experiment 2: SCM_RIGHTS FD Passing

**Hypothesis**: If we pass the GPU device file descriptors from the app to the helper via `SCM_RIGHTS` (Unix domain socket FD passing), the helper might be able to checkpoint using the app's FDs.

**What we tried**: Built `fd_pass_test.c`. App opens `/dev/nvidia0`, creates a CUDA context, allocates GPU memory, then passes the FD to a helper process via a Unix socket.

**Result**: `restore=401`. The driver doesn't care about the file descriptor — it tracks process identity internally. Having the same `struct file` doesn't help.

### Experiment 3: TID Matching via ns_last_pid

**Hypothesis**: Maybe the driver checks thread ID. If we make the helper's TID match the app's TID using `/proc/sys/kernel/ns_last_pid`, it might work.

**What we tried**: Built `tid_experiment.c`. Set `ns_last_pid` so the helper process gets the same TID as the original app thread that created the CUDA context.

**Result**: `restore=401`. TID matching doesn't help either. The driver uses some other internal identifier.

### Experiment 4: Driver 580 Migration API

**Hypothesis**: Driver 580 added `CUcheckpointRestoreArgs.gpuPairs` for GPU-to-GPU migration. Maybe this enables cross-process restore.

**What we tried**: Built `cross_proc_migration.c`. Used the migration API with GPU UUID pairs to restore on the same GPU.

**Result**: `restore=401`. The migration API is for same-process GPU swapping (e.g., moving from GPU 0 to GPU 1), not cross-process.

### Experiment 5: CRIU (Positive Control)

**Hypothesis**: If the driver requires same-process identity, CRIU should work because it restores the exact same process (same PID, same task_struct).

**What we tried**: Built `criu_positive_test.c`. Allocated GPU memory with pattern `0xDEADBEEF`, checkpointed with CRIU, restored.

**Result**: `restore=0`, pattern `0xDEADBEEF` verified. **CRIU works** because it restores the same process with the same identity. This confirmed that cross-process is the issue, not the API itself.

### Experiment 6: CRIU on gVisor Sentry

**Hypothesis**: Use CRIU to checkpoint the gVisor sentry process (which owns all GPU contexts via nvproxy).

**What we tried**: Ran `criu dump` targeting the sentry PID.

**Result**: CRIU failed. The sentry has `CLONE_VM` child threads that CRIU can't handle. gVisor's internal process model is incompatible with CRIU's expectations.

**Conclusion**: CRIU can't be used to checkpoint gVisor containers. Need to use gVisor's own checkpoint mechanism.

### Experiment 7: V100 (Volta) GPUs

**Hypothesis**: Test on V100 instances for broader GPU compatibility.

**Result**: `cuInit` returned `CUDA_ERROR_NO_DEVICE`. gVisor's nvproxy only supports Turing+ architectures (T4, A100, H100). V100/Volta is not supported. This is a gVisor limitation, not a CUDA limitation.

### Experiment 8: Docker-based Cold Restore

**Hypothesis**: Use Docker + runsc runtime for easier container lifecycle management.

**What we tried**: Configured Docker with runsc as the runtime, ran checkpoint via `docker checkpoint create`.

**Result**: Checkpoint succeeded, but cold restore failed. Docker cleans up the overlay rootfs when a container stops. When the sentry exits after checkpoint, Docker removes the filesystem, and there's nothing to restore into.

**Fix**: Use raw `runsc` with a self-contained OCI bundle. The bundle's rootfs persists on disk regardless of container state. This is also what Modal does — they manage rootfs directly.

---

## Experiments That Worked

### The Signal Handler Approach (The Solution)

**Hypothesis**: Since cross-process doesn't work, the app must checkpoint itself. Use gVisor's `--save-restore-exec-argv` to run a helper that signals the app, and the app calls cuda-checkpoint on `getpid()`.

**Implementation**:
1. App installs `SIGUSR1` → `on_checkpoint()` and `SIGUSR2` → `on_restore()` at startup
2. `on_checkpoint()` calls `cuCheckpointProcessLock(getpid(), args)` then `cuCheckpointProcessCheckpoint(getpid(), args)` — on ITSELF
3. `on_restore()` calls `cuCheckpointProcessRestore(getpid(), args)` then `cuCheckpointProcessUnlock(getpid(), args)`
4. Helper binary (`signal_helper.go`) sends the appropriate signal and waits for a completion flag

**Result**: `lock=0`, `checkpoint=0`, `restore=0`, `unlock=0`. GPU patterns intact after cold restore. This is the working solution.

### Single-GPU Cold Restore

**Test**: Allocate 4MB on GPU 0, fill with `0xBEEF1234`, checkpoint, kill sentry, restore on new sentry.

**Result**: PASS. `restore=0`, `val=0xBEEF1234` verified across 7+ consecutive ticks after restore. Checkpoint size: 21MB.

### Multi-GPU Cold Restore (2x H100)

**Test**: Allocate 4MB on each of 2 GPUs, fill GPU 0 with `0xCAFE0000` and GPU 1 with `0xCAFE0001`, checkpoint, kill sentry, restore.

**Result**: PASS. `restore=0`, both patterns verified across 6+ consecutive ticks. Checkpoint size: 36MB.

### Multi-GPU + NCCL Cold Restore

**Test**: Same as multi-GPU, plus initialize NCCL communicators, run allreduce across both GPUs, checkpoint, kill sentry, restore, verify patterns AND run allreduce again.

**Result**: PASS. `restore=0`, patterns intact, NCCL allreduce succeeds after restore. Checkpoint size: 632MB (includes NCCL internal GPU buffers).

**Key fix for NCCL**: gVisor's sandbox has no network interfaces by default. NCCL's bootstrap requires at least loopback. Fix:
```bash
ip netns add gvisor_ns
ip netns exec gvisor_ns ip link set lo up
```
Then reference the namespace in the OCI config: `{"type": "network", "path": "/var/run/netns/gvisor_ns"}`.

### PyTorch Model Checkpoint (--leave-running)

**Test**: Load a PyTorch `nn.Linear` model on 2x H100, create tensors on both GPUs, checkpoint with `--leave-running` (same sentry), restore.

**Result**: PASS. `restore=0`, tensors intact, model forward pass succeeds.

### Host Memory Proof

**Test**: `host_mem_test.c` — allocate 64MB on GPU, checkpoint, measure RSS change.

**Result**: RSS grew by 519MB after `cuCheckpointProcessCheckpoint`. Proves GPU state goes to host memory, not driver session. The 8x overhead includes page tables, context state, metadata.

---

## Infrastructure Issues

### Driver Version Not in gVisor

**Problem**: Lambda Labs instances had driver 570.148.08, which wasn't in gVisor's `version.go`. gVisor refused to start with `nvproxy-driver-version=570.148.08`.

**Fix**: Added one line to `pkg/sentry/devices/nvproxy/version.go`:
```go
_ = addDriverABI(570, 148, 8, "NO_DRIVER", "NO_DRIVER", v570_124_06)
```

Then rebuilt gVisor with Bazel (~2.5 minutes).

### CUDA Runtime vs Driver API in Raw runsc

**Problem**: Tests using CUDA runtime API (`cudaGetDeviceCount`, `cudaMalloc`) failed in raw runsc containers because the runtime requires additional shared libraries and initialization.

**Fix**: Use CUDA driver API via `dlopen("libcuda.so.1")` with function pointers (`cuInit`, `cuDeviceGetCount`, `cuMemAlloc_v2`, etc.). The driver API is lower-level but works directly through nvproxy.

### Rootfs Library Stack

**Problem**: Raw runsc containers need the full NVIDIA library chain in the rootfs. `libcuda.so.1` depends on `libnvidia-ptxjitcompiler.so.1`, which depends on `libnvidia-gpucomp.so.1`, which depends on `libnvidia-nvvm.so.4`.

**Fix**: Copy all NVIDIA libraries from the host into the container rootfs:
```bash
for lib in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-gpucomp.so.1 libnvidia-nvvm.so.4; do
    cp -L /usr/lib/x86_64-linux-gnu/$lib rootfs/lib/
done
```

### OCI Config Requirements

**Problem**: Initial OCI configs were missing the `/dev` tmpfs mount and explicit NVIDIA device entries. Without these, nvproxy can't create device files and `cuInit` returns `CUDA_ERROR_NO_DEVICE` (100).

**Working config requires**:
- `/dev` mounted as tmpfs (so nvproxy can create device nodes)
- Explicit device entries for `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`
- Resource access: `{"allow": true, "access": "rwm"}`
- For NCCL: network namespace with loopback

### GPU Scarcity on Lambda Labs

2x H100 SXM5 instances were extremely scarce. We monitored availability across regions `us-southeast-1`, `us-south-3`, `us-south-2` using the Lambda Labs API. Set up a polling routine to check `gpu_2x_h100_sxm5` capacity and auto-launch when available.

Lambda API: `https://cloud.lambdalabs.com/api/v1/instance-types` for availability, `/api/v1/instance-operations/launch` to create instances.

---

## Architecture of the Working Solution

```
HOST MACHINE
├── NVIDIA Driver (570.148.08)
├── GPUs: 2x H100 SXM5 (80GB HBM3 each, NVLink connected)
└── runsc (gVisor sentry)
    ├── nvproxy (intercepts GPU ioctls, forwards to host driver)
    ├── save-restore-exec (spawns helper binary during checkpoint/restore)
    └── Container Process (PID 1)
        ├── App code (gpu_snapshot.c)
        ├── Signal handlers (SIGUSR1 → checkpoint, SIGUSR2 → restore)
        ├── libcuda.so.1 (NVIDIA CUDA driver library)
        ├── CUDA contexts (one per GPU)
        ├── GPU memory allocations (tracked by driver)
        └── NCCL communicators (host-side state in process memory)
```

### Save Path

```
1. runsc checkpoint --save-restore-exec-argv=/bin/signal_helper
2. signal_helper spawns inside sandbox, mode=save
3. signal_helper sends SIGUSR1 to PID 1
4. on_checkpoint() fires:
   a. cuCheckpointProcessLock(getpid())
      - Driver gates all CUDA API calls
      - Drains in-flight GPU work on ALL GPUs
      - Both GPUs idle, atomically locked
   b. cuCheckpointProcessCheckpoint(getpid())
      - Driver DMA transfers VRAM → host memory
      - Records GPU virtual addresses
      - Saves context, stream, event, module state
      - GPU page table mappings saved
      - CUDA session terminated, GPUs released
      - RSS grows ~8x GPU allocation size
   c. Sets gpu_locked = 1
   d. Writes /tmp/.gpu_ckpt_done
5. signal_helper reads completion flag, exits
6. gVisor serializes entire process memory to disk
   - Includes anonymous pages with GPU state
   - Includes NCCL host-side state
   - Files: checkpoint.img, pages.img, pages_meta.img
7. Sentry exits (cold checkpoint)
```

### Restore Path

```
1. runsc restore --image-path=/tmp/ckpt
2. NEW sentry starts, new PID, new driver session
3. nvproxy opens fresh /dev/nvidia* FDs on host
4. gVisor loads checkpoint files:
   - Process memory restored (includes GPU state in host pages)
   - File descriptors recreated
   - Signal handlers restored
   - CPU registers restored, threads resume
5. App wakes, sees gpu_locked=1, prints "gpu_locked"
6. signal_helper spawns, mode=restore
7. signal_helper sends SIGUSR2 to PID 1
8. on_restore() fires:
   a. cuCheckpointProcessRestore(getpid())
      - Driver reads GPU state from host memory
      - Allocates GPU memory at ORIGINAL virtual addresses
      - DMA transfers host → VRAM
      - Rebuilds GPU page tables
      - Recreates contexts, streams, events, modules
      - New CUDA session, but same data
   b. cuCheckpointProcessUnlock(getpid())
      - Driver opens API gate
      - CUDA calls allowed again
   c. Sets gpu_locked = 0
   d. Writes /tmp/.gpu_ckpt_done
9. App continues — all pointers valid, all data intact
```

---

## What the cuda-checkpoint API Captures

### Captured
- Device memory contents (all cuMemAlloc allocations)
- GPU virtual address mappings (pointers remain valid after restore)
- CUDA contexts and configuration
- Stream objects and ordering
- Event state (recorded/completed)
- Loaded PTX/cubin modules
- GPU page table entries

### NOT Captured
- Unified Virtual Memory (UVM) allocations — unsupported
- IPC memory (peer-GPU shared allocations) — unsupported
- In-flight kernel execution state — kernels must complete before checkpoint
- Live GPU register state — because kernels already finished

### Requirements
- NVIDIA driver 550+ (basic), 570+ (timeout, NVML), 580+ (GPU migration)
- x86_64 only
- Linux only
- Same GPU type for restore (GPU virtual address space must match)
- Persistence mode recommended (`nvidia-smi -pm 1`)

---

## Driver Session Binding (Why Cross-Process Fails)

The NVIDIA driver binds checkpoint data to an internal process identity structure. This is NOT the PID, NOT the file descriptor, NOT the TID. It appears to be a driver-internal handle tied to the process's initial `cuInit` call.

| Approach | Result |
|----------|--------|
| Same `struct file` via SCM_RIGHTS | restore=401 |
| Same TID via ns_last_pid | restore=401 |
| Persistence mode ON | restore=401 |
| Driver 580 migration API | restore=401 |
| CRIU (same PID + task_struct + everything) | restore=0 ✓ |
| Self-checkpoint (getpid()) | restore=0 ✓ |

Only two things work: CRIU (which restores the exact process identity) and self-checkpoint (the process itself calls the API). Our solution uses self-checkpoint via signal handlers.

---

## Verified Test Results

### Session 1 (driver 580.105.08, 2x H100)

| Test | Type | Sentry | Result |
|------|------|--------|--------|
| Single-GPU C test | --leave-running | Same | restore=0, 0xBEEF0001 ✓ |
| Multi-GPU C test | --leave-running | Same | restore=0, 0xCAFE0000+0xCAFE0001 ✓ |
| PyTorch nn.Linear | --leave-running | Same | restore=0, tensors+model ✓ |
| Single-GPU cold | Raw runsc | NEW | restore=0, 0xBEEF1234, 15 ticks ✓ |
| Multi-GPU cold | Raw runsc | NEW | restore=0, 0xCAFE0000+0xCAFE0001, 15 ticks ✓ |
| Multi-GPU+NCCL cold | Raw runsc | NEW | restore=0, patterns+allreduce, 14 ticks ✓ |

### Session 2 (driver 570.148.08, 2x H100)

| Test | Type | Sentry | Checkpoint Size | Result |
|------|------|--------|----------------|--------|
| Single-GPU cold | Raw runsc | NEW | 21MB | restore=0, 0xBEEF1234, 7+ ticks ✓ |
| Multi-GPU cold | Raw runsc | NEW | 36MB | restore=0, 0xCAFE0000+0xCAFE0001, 6+ ticks ✓ |
| Multi-GPU+NCCL cold | Raw runsc | NEW | 632MB | restore=0, unlock=0, running ✓ |

---

## Negative Results (Important)

| What | Why It Doesn't Work |
|------|-------------------|
| V100/Volta GPUs | gVisor nvproxy only supports Turing+ |
| Docker cold restore | Docker cleans up overlay rootfs on container stop |
| Cross-process checkpoint | NVIDIA driver binds to process identity (401) |
| CRIU on gVisor sentry | CLONE_VM children block CRIU dump |
| CUDA runtime API in raw runsc | Requires additional libs/init; use driver API |
| cuMemcpyDtoH without /dev tmpfs mount | cuInit returns 100 (NO_DEVICE) |

---

## Comparison with Modal

Modal's [GPU Memory Snapshots blog](https://modal.com/blog/gpu-mem-snapshots) confirms the same approach:

> The driver checkpoints device memory in host memory so that it can be checkpointed to disk by a host-side checkpointing system. Then, once the host-side system has restored the host memory, including the device checkpoint, the driver restores the device memory.

Our implementation matches this exactly. The differences are operational:
- Modal has automated rootfs management (we use manual OCI bundles)
- Modal injects checkpoint handlers transparently (our apps must opt in)
- Modal has a warm pool for fast restore (we cold-start every time)
- Modal manages GPU scheduling and cross-machine migration (we're single-machine)

The core mechanism — cuda-checkpoint self-checkpoint via signal handlers, gVisor serializes host memory, cold restore on new sentry — is identical.

---

## Pending / Untested

- **Cross-machine restore**: Same mechanism should work (checkpoint file is portable if same GPU type), but untested on different physical machines
- **Large models**: Tested with 4MB allocations and nn.Linear, not with real LLMs (70B+ params, tens of GB VRAM)
- **GPUDirect RDMA**: NCCL tested over loopback TCP. Production NCCL uses RDMA over InfiniBand — unknown if that transport state survives checkpoint
- **NVLink direct transport**: NCCL may use NVLink directly for intra-node communication — untested whether that state is captured
- **LD_PRELOAD injection**: Building a shared library that auto-installs signal handlers so apps don't need code changes

---

## Lambda Labs Instances Used

All testing done on Lambda Labs cloud instances:

- GPU types tested: 2x H100 SXM5 (80GB HBM3), 1x H100, 1x A100, 1x T4, 1x V100
- Regions: us-south-3, us-southeast-1, us-south-2
- Driver versions encountered: 570.148.08, 580.105.08
- SSH key: "gvisor-test"
- API: `https://cloud.lambdalabs.com/api/v1/`

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `gpu_snapshot.c` | The complete solution — signal handlers, cuda-checkpoint API loading, example app |
| `signal_helper.go` | gVisor save-restore-exec helper binary |
| `README.md` | Usage guide and test results |
| `RESEARCH.md` | This file — full research log |
| `docscrawl.py` | Documentation crawler tool (unrelated to GPU snapshots) |

## Files in the Original Repo (gpu-checkpoint-gvisor)

| File | Purpose |
|------|---------|
| `gvisor-nvproxy-checkpoint.patch` | Base gVisor patch for checkpoint support |
| `test/gpu_checkpoint_test.c` | Working checkpoint test (signal-based) |
| `test/cold_restore_test.sh` | Cold restore test script |
| `test/multi_gpu_nccl_cold.sh` | Multi-GPU NCCL cold restore script |
| `test/torch_checkpoint_test.py` | PyTorch model checkpoint test |
| `test/host_mem_test.c` | Host memory proof (RSS growth) |
| `test/criu_positive_test.c` | CRIU restore proof |
| `test/tid_experiment.c` | TID matching experiment (401) |
| `test/fd_pass_test.c` | SCM_RIGHTS experiment (401) |
| `test/cross_proc_migration.c` | Driver 580 migration experiment (401) |
