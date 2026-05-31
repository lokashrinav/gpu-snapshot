// signal_helper.go — Save-restore-exec helper for gVisor GPU checkpoint.
//
// This binary runs inside the sandbox via gVisor's --save-restore-exec-argv.
// It sends SIGUSR1 (save) or SIGUSR2 (restore) to PID 1 and waits for
// the app to complete the cuda-checkpoint operation.
//
// Build: go build -o signal_helper signal_helper.go
package main

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"time"
)

func main() {
	mode := os.Getenv("GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE")
	if mode == "" {
		fmt.Fprintln(os.Stderr, "signal_helper: GVISOR_SAVE_RESTORE_AUTO_EXEC_MODE not set")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "signal_helper: mode=%s\n", mode)

	var sig syscall.Signal
	switch mode {
	case "save":
		sig = syscall.SIGUSR1
	case "restore", "resume":
		sig = syscall.SIGUSR2
	default:
		fmt.Fprintf(os.Stderr, "signal_helper: unknown mode %q\n", mode)
		os.Exit(1)
	}

	os.Remove("/tmp/.gpu_ckpt_done")

	if err := syscall.Kill(1, sig); err != nil {
		fmt.Fprintf(os.Stderr, "signal_helper: kill(1, %d): %v\n", sig, err)
		os.Exit(1)
	}

	for i := 0; i < 120; i++ {
		if d, err := os.ReadFile("/tmp/.gpu_ckpt_done"); err == nil {
			rc := strings.TrimSpace(string(d))
			fmt.Fprintf(os.Stderr, "signal_helper: %s done rc=%s\n", mode, rc)
			if rc != "0" {
				os.Exit(1)
			}
			return
		}
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Fprintln(os.Stderr, "signal_helper: timed out waiting for app")
	os.Exit(1)
}
