/*
 * GPU Snapshot — Multi-GPU checkpoint/restore for gVisor containers.
 *
 * The app calls cuda-checkpoint on itself via signal handlers.
 * gVisor serializes process memory (which contains GPU state after
 * checkpoint) to disk. On restore, the app reads GPU state from
 * its own host memory.
 *
 * Build: gcc gpu_snapshot.c -o gpu_snapshot -I/usr/local/cuda/include -ldl
 *        (needs CUDA toolkit headers installed)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <dlfcn.h>
#include <cuda.h>

static CUresult (*checkpoint_lock)(int, CUcheckpointLockArgs *);
static CUresult (*checkpoint_checkpoint)(int, CUcheckpointCheckpointArgs *);
static CUresult (*checkpoint_restore)(int, CUcheckpointRestoreArgs *);
static CUresult (*checkpoint_unlock)(int, CUcheckpointUnlockArgs *);

static volatile sig_atomic_t gpu_locked = 0;

void on_checkpoint(int sig) {
    int pid = getpid();

    CUcheckpointLockArgs lock_args;
    memset(&lock_args, 0, sizeof(lock_args));
    CUresult r = checkpoint_lock(pid, &lock_args);
    fprintf(stderr, "SAVE: lock=%d\n", r);

    CUcheckpointCheckpointArgs ckpt_args;
    memset(&ckpt_args, 0, sizeof(ckpt_args));
    r = checkpoint_checkpoint(pid, &ckpt_args);
    fprintf(stderr, "SAVE: checkpoint=%d\n", r);

    gpu_locked = 1;

    FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
}

void on_restore(int sig) {
    int pid = getpid();

    CUcheckpointRestoreArgs restore_args;
    memset(&restore_args, 0, sizeof(restore_args));
    fprintf(stderr, "RESTORE: restore(%d)\n", pid);
    CUresult r = checkpoint_restore(pid, &restore_args);
    fprintf(stderr, "RESTORE: restore=%d\n", r);

    if (r == CUDA_SUCCESS) {
        CUcheckpointUnlockArgs unlock_args;
        memset(&unlock_args, 0, sizeof(unlock_args));
        r = checkpoint_unlock(pid, &unlock_args);
        fprintf(stderr, "RESTORE: unlock=%d\n", r);
    } else {
        fprintf(stderr, "RESTORE: FAILED=%d\n", r);
    }

    gpu_locked = 0;

    FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
}

int load_checkpoint_api(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) {
        fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror());
        return -1;
    }
    checkpoint_lock = dlsym(h, "cuCheckpointProcessLock");
    checkpoint_checkpoint = dlsym(h, "cuCheckpointProcessCheckpoint");
    checkpoint_restore = dlsym(h, "cuCheckpointProcessRestore");
    checkpoint_unlock = dlsym(h, "cuCheckpointProcessUnlock");
    if (!checkpoint_lock || !checkpoint_checkpoint ||
        !checkpoint_restore || !checkpoint_unlock) {
        fprintf(stderr, "cuda-checkpoint symbols not found (need driver 570+)\n");
        return -1;
    }
    return 0;
}

void install_checkpoint_handlers(void) {
    if (load_checkpoint_api() != 0) {
        fprintf(stderr, "WARNING: cuda-checkpoint not available\n");
        return;
    }
    signal(SIGUSR1, on_checkpoint);
    signal(SIGUSR2, on_restore);
}

int is_gpu_locked(void) {
    return gpu_locked;
}

/* --- Example usage below --- */

int main(void) {
    install_checkpoint_handlers();

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
        cuDevicePrimaryCtxRetain(&ctx[i], dev);
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
