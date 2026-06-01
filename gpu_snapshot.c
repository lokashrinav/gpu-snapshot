/*
 * GPU Snapshot — Multi-GPU checkpoint/restore for gVisor containers.
 *
 * The app calls cuda-checkpoint on itself via signal handlers.
 * gVisor serializes process memory (which contains GPU state after
 * checkpoint) to disk. On restore, the app reads GPU state from
 * its own host memory.
 *
 * Build: gcc gpu_snapshot.c -o gpu_snapshot -I/usr/local/cuda/include -lcuda
 *        (needs CUDA toolkit headers and driver 570+)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <cuda.h>

static volatile sig_atomic_t gpu_locked = 0;

void on_checkpoint(int sig) {
    int pid = getpid();

    CUcheckpointLockArgs lock_args;
    memset(&lock_args, 0, sizeof(lock_args));
    CUresult r = cuCheckpointProcessLock(pid, &lock_args);
    fprintf(stderr, "SAVE: lock=%d\n", r);

    CUcheckpointCheckpointArgs ckpt_args;
    memset(&ckpt_args, 0, sizeof(ckpt_args));
    r = cuCheckpointProcessCheckpoint(pid, &ckpt_args);
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
    CUresult r = cuCheckpointProcessRestore(pid, &restore_args);
    fprintf(stderr, "RESTORE: restore=%d\n", r);

    if (r == CUDA_SUCCESS) {
        CUcheckpointUnlockArgs unlock_args;
        memset(&unlock_args, 0, sizeof(unlock_args));
        r = cuCheckpointProcessUnlock(pid, &unlock_args);
        fprintf(stderr, "RESTORE: unlock=%d\n", r);
    } else {
        fprintf(stderr, "RESTORE: FAILED=%d\n", r);
    }

    gpu_locked = 0;

    FILE *f = fopen("/tmp/.gpu_ckpt_done", "w");
    if (f) { fprintf(f, "0\n"); fclose(f); }
}

void install_checkpoint_handlers(void) {
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
