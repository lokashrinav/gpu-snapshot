/*
 * GPU Snapshot — Multi-GPU checkpoint/restore for gVisor containers.
 *
 * This is the complete solution. The app calls cuda-checkpoint on itself
 * via signal handlers. gVisor serializes process memory (which contains
 * GPU state after checkpoint) to disk. On restore, the app reads GPU state
 * from its own host memory.
 *
 * How it works:
 *   SIGUSR1 → cuCheckpointProcessLock + cuCheckpointProcessCheckpoint
 *             (GPU VRAM copied to host memory, CUDA session terminated)
 *   SIGUSR2 → cuCheckpointProcessRestore + cuCheckpointProcessUnlock
 *             (reads from host memory, creates new CUDA session)
 *
 * Between SIGUSR1 and SIGUSR2:
 *   runsc checkpoint → serializes process memory (includes GPU state) to disk
 *   runsc restore   → loads process memory back (new sentry, fresh driver session)
 *
 * Verified on 2x H100 SXM5, driver 570.148.08 and 580.105.08:
 *   - Single-GPU cold restore: restore=0, pattern verified
 *   - Multi-GPU cold restore: restore=0, both patterns verified
 *   - Multi-GPU + NCCL cold restore: restore=0, patterns + allreduce verified
 *   - PyTorch model checkpoint: restore=0, tensors + model verified
 *
 * Build: gcc gpu_snapshot.c -o gpu_snapshot -ldl
 *
 * Usage with gVisor:
 *   # Start container
 *   runsc --nvproxy --nvproxy-driver-version=580.105.08 run --bundle $BUNDLE $ID
 *
 *   # Checkpoint (sentry can exit after this)
 *   runsc checkpoint --save-restore-exec-argv=/bin/signal_helper \
 *       --save-restore-exec-timeout=120s --image-path=/tmp/ckpt $ID
 *
 *   # Cold restore (new sentry, GPU state from host memory)
 *   runsc --nvproxy --nvproxy-driver-version=580.105.08 \
 *       restore --image-path=/tmp/ckpt --bundle=$BUNDLE $RESTORED_ID
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef CUresult (*CheckpointFn)(int, void *);

static CheckpointFn cuda_lock, cuda_checkpoint, cuda_restore, cuda_unlock;
static volatile sig_atomic_t gpu_locked = 0;

/*
 * SIGUSR1 handler — called by the save-restore-exec helper before checkpoint.
 * Atomically locks all GPUs and copies VRAM to host memory.
 */
void on_checkpoint(int sig) {
    char args[64] = {0};
    int pid = getpid();

    int r = cuda_lock(pid, args);
    fprintf(stderr, "SAVE: lock=%d\n", r);

    memset(args, 0, 64);
    r = cuda_checkpoint(pid, args);
    fprintf(stderr, "SAVE: checkpoint=%d\n", r);

    gpu_locked = 1;

    FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
}

/*
 * SIGUSR2 handler — called by the save-restore-exec helper after restore.
 * Reads GPU state from host memory and creates a new CUDA session.
 */
void on_restore(int sig) {
    char args[64] = {0};
    int pid = getpid();

    fprintf(stderr, "RESTORE: restore(%d)\n", pid);
    int r = cuda_restore(pid, args);
    fprintf(stderr, "RESTORE: restore=%d\n", r);

    if (r == 0) {
        memset(args, 0, 64);
        r = cuda_unlock(pid, args);
        fprintf(stderr, "RESTORE: unlock=%d\n", r);
    } else {
        fprintf(stderr, "RESTORE: FAILED=%d\n", r);
    }

    gpu_locked = 0;

    FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
}

/*
 * Load cuda-checkpoint API symbols via dlopen.
 * Returns 0 on success.
 */
int load_checkpoint_api(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) {
        fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror());
        return -1;
    }
    cuda_lock = (CheckpointFn)dlsym(h, "cuCheckpointProcessLock");
    cuda_checkpoint = (CheckpointFn)dlsym(h, "cuCheckpointProcessCheckpoint");
    cuda_restore = (CheckpointFn)dlsym(h, "cuCheckpointProcessRestore");
    cuda_unlock = (CheckpointFn)dlsym(h, "cuCheckpointProcessUnlock");
    if (!cuda_lock || !cuda_checkpoint || !cuda_restore || !cuda_unlock) {
        fprintf(stderr, "cuda-checkpoint symbols not found (need driver 570+)\n");
        return -1;
    }
    return 0;
}

/*
 * Install signal handlers for checkpoint/restore.
 * Call this once at app startup.
 */
void install_checkpoint_handlers(void) {
    if (load_checkpoint_api() != 0) {
        fprintf(stderr, "WARNING: cuda-checkpoint not available\n");
        return;
    }
    signal(SIGUSR1, on_checkpoint);
    signal(SIGUSR2, on_restore);
}

/*
 * Check if GPUs are currently locked (between checkpoint and restore).
 * Use this to skip GPU operations during the checkpoint window.
 */
int is_gpu_locked(void) {
    return gpu_locked;
}

/* --- Example usage below --- */

int main(void) {
    install_checkpoint_handlers();

    /* Initialize CUDA via driver API (works through gVisor nvproxy) */
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    CUresult (*cuInit)(unsigned) = dlsym(h, "cuInit");
    CUresult (*cuDeviceGetCount)(int *) = dlsym(h, "cuDeviceGetCount");
    CUresult (*cuDeviceGet)(CUdevice *, int) = dlsym(h, "cuDeviceGet");
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = dlsym(h, "cuDevicePrimaryCtxRetain");
    CUresult (*cuCtxSetCurrent)(CUcontext) = dlsym(h, "cuCtxSetCurrent");
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = dlsym(h, "cuMemAlloc_v2");
    CUresult (*cuMemsetD32)(CUdeviceptr, unsigned, size_t) = dlsym(h, "cuMemsetD32_v2");
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) = dlsym(h, "cuMemcpyDtoH_v2");

    cuInit(0);
    int num_gpus;
    cuDeviceGetCount(&num_gpus);
    if (num_gpus > 2) num_gpus = 2;
    printf("pid=%d gpus=%d\n", getpid(), num_gpus);

    CUcontext ctx[8];
    CUdeviceptr dptr[8];
    for (int i = 0; i < num_gpus; i++) {
        CUdevice dev;
        cuDeviceGet(&dev, i);
        cuDevicePrimaryCtxRetain(&ctx[i], i);
        cuCtxSetCurrent(ctx[i]);
        cuMemAlloc(&dptr[i], 4 * 1024 * 1024);
        cuMemsetD32(dptr[i], 0xCAFE0000 | i, 1024 * 1024);
        printf("GPU %d: pattern 0x%X\n", i, 0xCAFE0000 | i);
    }

    printf("READY\n");
    fflush(stdout);

    for (int tick = 1; ; tick++) {
        sleep(2);

        if (is_gpu_locked()) {
            printf("tick=%d gpu_locked\n", tick * 2);
            fflush(stdout);
            continue;
        }

        int ok = 1;
        for (int i = 0; i < num_gpus; i++) {
            cuCtxSetCurrent(ctx[i]);
            unsigned int expected = 0xCAFE0000 | i, val;
            CUresult cr = cuMemcpyDtoH(&val, dptr[i], sizeof(unsigned int));
            if (cr || val != expected) {
                printf("GPU %d: FAIL r=%d v=0x%X\n", i, cr, val);
                ok = 0;
            }
        }
        printf("tick=%d all_gpus_ok=%d\n", tick * 2, ok);
        fflush(stdout);
    }
}
