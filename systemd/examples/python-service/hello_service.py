# © 2025 [Supun Akalanka Sriyananda] — MIT License
"""
hello_service.py
----------------
A minimal long-running Python service for learning systemd.
Logs a heartbeat message every 10 seconds so you can watch it
in real time with: journalctl -u hello-python.service -f

This script intentionally does nothing useful — it's designed
purely to demonstrate the lifecycle of a process that systemd
starts, monitors, and restarts if it dies.

Deploy:
  sudo cp hello-python.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now hello-python.service
"""

import logging
import signal
import sys
import time

# Write logs to stdout so systemd's journal captures them automatically.
# The format includes a timestamp so log entries are self-contained
# even if you're reading them in a raw log file rather than journalctl.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Track whether a shutdown has been requested.
# This lets the main loop finish its current work cleanly
# rather than being interrupted mid-task by a kill signal.
shutdown_requested = False


def handle_shutdown(signum, frame):
    """Called by systemd when it sends SIGTERM (systemctl stop)."""
    global shutdown_requested
    logger.info(f"Received signal {signum} — preparing to shut down gracefully")
    shutdown_requested = True


def main():
    # Register the graceful shutdown handler for SIGTERM.
    # When you run 'systemctl stop', systemd sends SIGTERM first,
    # giving the process a chance to clean up. If we don't handle it,
    # the process is forcibly killed after TimeoutStopSec seconds.
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Service starting up successfully")
    counter = 0

    while not shutdown_requested:
        counter += 1
        logger.info(f"Heartbeat #{counter} — service is alive and healthy")

        # Sleep in short increments so we can check shutdown_requested
        # frequently, making shutdown feel responsive rather than
        # waiting up to 10 seconds for the current sleep to finish.
        for _ in range(10):
            if shutdown_requested:
                break
            time.sleep(1)

    logger.info(f"Clean shutdown after {counter} heartbeat(s). Goodbye.")
    sys.exit(0)


if __name__ == "__main__":
    main()
