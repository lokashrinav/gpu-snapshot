#!/bin/bash
#
# GPU Snapshot — Setup and run script.
#
# This script does everything:
#   1. Builds gVisor from source
#   2. Compiles gpu_snapshot and signal_helper
#   3. Creates the OCI bundle (rootfs + config)
#   4. Sets up networking for NCCL
#   5. Starts the container
#
# Usage:
#   ./setup.sh build     — compile everything
#   ./setup.sh run       — start the container
#   ./setup.sh checkpoint — checkpoint the container
#   ./setup.sh restore   — cold restore from checkpoint
#   ./setup.sh all       — build + run (then checkpoint/restore manually)
#
# Requirements:
#   - Linux with NVIDIA GPUs (Turing+)
#   - NVIDIA driver 570+
#   - CUDA toolkit installed (for headers)
#   - Go 1.21+ (for gVisor and signal_helper)
#   - gcc
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GVISOR_DIR="${GVISOR_DIR:-/tmp/gvisor}"
BUNDLE_DIR="${BUNDLE_DIR:-/tmp/gpu-snapshot-bundle}"
CKPT_DIR="${CKPT_DIR:-/tmp/gpu-snapshot-checkpoint}"
CONTAINER_ID="${CONTAINER_ID:-gpu-snapshot}"
RESTORED_ID="${RESTORED_ID:-gpu-snapshot-restored}"

# Auto-detect driver version
detect_driver_version() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
}

DRIVER_VERSION="${DRIVER_VERSION:-$(detect_driver_version 2>/dev/null || echo "570.148.08")}"

# Auto-detect GPU count
detect_gpu_count() {
    nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
}

GPU_COUNT="${GPU_COUNT:-$(detect_gpu_count 2>/dev/null || echo "1")}"

# Find runsc binary
find_runsc() {
    if command -v runsc &>/dev/null; then
        echo "runsc"
    elif [ -f "$GVISOR_DIR/bazel-bin/runsc/runsc_/runsc" ]; then
        echo "$GVISOR_DIR/bazel-bin/runsc/runsc_/runsc"
    else
        # Search in bazel cache
        find /home -name "runsc" -path "*/bazel-out/*/bin/runsc/runsc_/runsc" 2>/dev/null | head -1
    fi
}

# -------------------------------------------------------
# BUILD
# -------------------------------------------------------
cmd_build() {
    echo "=== Building GPU Snapshot ==="
    echo "Driver: $DRIVER_VERSION"
    echo "GPUs:   $GPU_COUNT"
    echo ""

    # Build gVisor if runsc not found
    RUNSC=$(find_runsc)
    if [ -z "$RUNSC" ]; then
        echo "[1/4] Building gVisor from source..."
        if [ ! -d "$GVISOR_DIR" ]; then
            git clone https://github.com/google/gvisor.git "$GVISOR_DIR"
        fi
        cd "$GVISOR_DIR"
        bazel build //runsc
        RUNSC="$GVISOR_DIR/bazel-bin/runsc/runsc_/runsc"
        echo "Built: $RUNSC"
    else
        echo "[1/4] gVisor found: $RUNSC"
    fi

    echo ""

    # Compile gpu_snapshot
    echo "[2/4] Compiling gpu_snapshot..."
    CUDA_INCLUDE=""
    for dir in /usr/local/cuda/include /usr/include /usr/local/cuda-*/include; do
        if [ -f "$dir/cuda.h" ]; then
            CUDA_INCLUDE="-I$dir"
            break
        fi
    done
    if [ -z "$CUDA_INCLUDE" ]; then
        echo "ERROR: cuda.h not found. Install CUDA toolkit."
        exit 1
    fi
    gcc "$SCRIPT_DIR/gpu_snapshot.c" -o "$SCRIPT_DIR/gpu_snapshot" $CUDA_INCLUDE -lcuda
    echo "Built: $SCRIPT_DIR/gpu_snapshot"

    # Compile signal_helper
    echo "[3/4] Compiling signal_helper..."
    cd "$SCRIPT_DIR"
    CGO_ENABLED=0 go build -o signal_helper signal_helper.go
    echo "Built: $SCRIPT_DIR/signal_helper"

    # Create OCI bundle
    echo "[4/4] Creating OCI bundle..."
    create_bundle

    echo ""
    echo "=== Build complete ==="
    echo "Bundle: $BUNDLE_DIR"
    echo "Run:    $0 run"
}

# -------------------------------------------------------
# CREATE OCI BUNDLE
# -------------------------------------------------------
create_bundle() {
    rm -rf "$BUNDLE_DIR"
    mkdir -p "$BUNDLE_DIR/rootfs"/{bin,lib,lib64,tmp,proc,sys,dev,etc}

    # Copy binaries
    cp "$SCRIPT_DIR/gpu_snapshot" "$BUNDLE_DIR/rootfs/bin/"
    cp "$SCRIPT_DIR/signal_helper" "$BUNDLE_DIR/rootfs/bin/"
    cp /bin/sh "$BUNDLE_DIR/rootfs/bin/" 2>/dev/null || true

    # Copy NVIDIA libraries
    for lib in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-gpucomp.so.1 libnvidia-nvvm.so.4; do
        src=$(find /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/lib64/ /usr/lib64/ -name "${lib}*" 2>/dev/null | head -1)
        if [ -n "$src" ]; then
            cp -L "$src" "$BUNDLE_DIR/rootfs/lib/"
        fi
    done

    # Copy libc and dynamic linker
    cp /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$BUNDLE_DIR/rootfs/lib64/" 2>/dev/null || true
    for lib in libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1; do
        src=$(find /lib/x86_64-linux-gnu/ -name "$lib" 2>/dev/null | head -1)
        [ -n "$src" ] && cp -L "$src" "$BUNDLE_DIR/rootfs/lib/"
    done

    echo "nameserver 8.8.8.8" > "$BUNDLE_DIR/rootfs/etc/resolv.conf"

    # Build device list based on GPU count
    DEVICES=""
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        DEVICES="$DEVICES{\"path\": \"/dev/nvidia$i\", \"type\": \"c\", \"major\": 195, \"minor\": $i, \"fileMode\": 438},"
    done

    # Set up network namespace for NCCL
    if ! ip netns list 2>/dev/null | grep -q gvisor_ns; then
        ip netns add gvisor_ns
        ip netns exec gvisor_ns ip link set lo up
    fi

    # Write OCI config
    cat > "$BUNDLE_DIR/config.json" << JSONEOF
{
  "ociVersion": "1.0.0",
  "process": {
    "user": {"uid": 0, "gid": 0},
    "args": ["/bin/gpu_snapshot"],
    "env": [
      "PATH=/bin",
      "LD_LIBRARY_PATH=/lib64:/lib:/lib/x86_64-linux-gnu",
      "NCCL_SOCKET_IFNAME=lo",
      "NCCL_DEBUG=WARN"
    ],
    "cwd": "/"
  },
  "root": {"path": "rootfs", "readonly": false},
  "mounts": [
    {"destination": "/proc", "type": "proc", "source": "proc"},
    {"destination": "/dev", "type": "tmpfs", "source": "tmpfs"},
    {"destination": "/sys", "type": "sysfs", "source": "sysfs",
     "options": ["nosuid", "noexec", "nodev", "ro"]},
    {"destination": "/tmp", "type": "tmpfs", "source": "none"}
  ],
  "linux": {
    "namespaces": [
      {"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"},
      {"type": "network", "path": "/var/run/netns/gvisor_ns"}
    ],
    "devices": [
      $DEVICES
      {"path": "/dev/nvidiactl", "type": "c", "major": 195, "minor": 255, "fileMode": 438},
      {"path": "/dev/nvidia-uvm", "type": "c", "major": 234, "minor": 0, "fileMode": 438},
      {"path": "/dev/nvidia-uvm-tools", "type": "c", "major": 234, "minor": 1, "fileMode": 438}
    ],
    "resources": {"devices": [{"allow": true, "access": "rwm"}]}
  }
}
JSONEOF

    echo "Bundle created: $BUNDLE_DIR"
}

# -------------------------------------------------------
# RUN
# -------------------------------------------------------
cmd_run() {
    RUNSC=$(find_runsc)
    if [ -z "$RUNSC" ]; then
        echo "ERROR: runsc not found. Run '$0 build' first."
        exit 1
    fi

    # Enable persistence mode
    nvidia-smi -pm 1 2>/dev/null || true

    # Clean up old container
    "$RUNSC" --rootless=false delete "$CONTAINER_ID" 2>/dev/null || true

    echo "=== Starting container ==="
    echo "runsc:  $RUNSC"
    echo "driver: $DRIVER_VERSION"
    echo "bundle: $BUNDLE_DIR"
    echo ""

    "$RUNSC" --nvproxy --nvproxy-driver-version="$DRIVER_VERSION" \
        --rootless=false \
        run --bundle "$BUNDLE_DIR" "$CONTAINER_ID"
}

# -------------------------------------------------------
# CHECKPOINT
# -------------------------------------------------------
cmd_checkpoint() {
    RUNSC=$(find_runsc)
    if [ -z "$RUNSC" ]; then
        echo "ERROR: runsc not found."
        exit 1
    fi

    rm -rf "$CKPT_DIR"
    mkdir -p "$CKPT_DIR"

    echo "=== Checkpointing container ==="
    "$RUNSC" --rootless=false checkpoint \
        --save-restore-exec-argv=/bin/signal_helper \
        --save-restore-exec-timeout=120s \
        --image-path="$CKPT_DIR" \
        "$CONTAINER_ID"

    echo ""
    echo "Checkpoint saved: $CKPT_DIR ($(du -sh "$CKPT_DIR" | cut -f1))"
    echo "Container stopped. Run '$0 restore' to cold restore."
}

# -------------------------------------------------------
# RESTORE
# -------------------------------------------------------
cmd_restore() {
    RUNSC=$(find_runsc)
    if [ -z "$RUNSC" ]; then
        echo "ERROR: runsc not found."
        exit 1
    fi

    # Clean up old restored container
    "$RUNSC" --rootless=false delete "$RESTORED_ID" 2>/dev/null || true

    echo "=== Cold restoring container ==="
    echo "Checkpoint: $CKPT_DIR"
    echo ""

    "$RUNSC" --nvproxy --nvproxy-driver-version="$DRIVER_VERSION" \
        --rootless=false \
        restore \
        --image-path="$CKPT_DIR" \
        --bundle="$BUNDLE_DIR" \
        "$RESTORED_ID"
}

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
case "${1:-}" in
    build)
        cmd_build
        ;;
    run)
        cmd_run
        ;;
    checkpoint)
        cmd_checkpoint
        ;;
    restore)
        cmd_restore
        ;;
    all)
        cmd_build
        echo ""
        cmd_run
        ;;
    *)
        echo "GPU Snapshot — Multi-GPU checkpoint/restore for gVisor"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  build       Compile everything and create OCI bundle"
        echo "  run         Start the container"
        echo "  checkpoint  Checkpoint the running container (sentry exits)"
        echo "  restore     Cold restore from checkpoint (new sentry)"
        echo "  all         Build + run"
        echo ""
        echo "Typical flow:"
        echo "  $0 build              # compile and set up"
        echo "  $0 run &              # start container in background"
        echo "  # ... wait for app to be ready ..."
        echo "  $0 checkpoint         # save and stop"
        echo "  $0 restore            # bring it back"
        ;;
esac
