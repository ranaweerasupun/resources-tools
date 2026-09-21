# © 2025 [Supun Akalanka Sriyananda] — MIT License
"""
cleanup.py
----------
A maintenance script designed to be run on a schedule by a systemd timer.
Deletes files older than MAX_AGE_DAYS from TARGET_DIR.

This is a realistic example of a periodic task that many embedded devices
need — clearing old logs, sensor data, or temporary files to prevent
storage from filling up over time.

Key design points for a timer-triggered script:
  - It runs once and exits (does not loop forever like a long-running service)
  - It logs clearly so you can see what happened in: journalctl -u cleanup.service
  - It exits with code 0 on success and non-zero on partial failure, so
    systemd can accurately report whether the task succeeded

Deploy:
  sudo cp cleanup.service /etc/systemd/system/
  sudo cp cleanup.timer   /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now cleanup.timer

Check scheduled runs:
  systemctl list-timers

Trigger manually to test:
  sudo systemctl start cleanup.service
  journalctl -u cleanup.service -n 30
"""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Read configuration from environment variables set in the service file.
# Defaults are provided so the script works if run manually without systemd.
TARGET_DIR   = os.environ.get("CLEANUP_DIR",    "/home/pi/data")
MAX_AGE_DAYS = int(os.environ.get("MAX_AGE_DAYS", "7"))


def main():
    logger.info(f"Starting cleanup: removing files older than {MAX_AGE_DAYS} day(s) from '{TARGET_DIR}'")

    if not os.path.isdir(TARGET_DIR):
        logger.warning(f"Target directory does not exist: {TARGET_DIR}")
        logger.warning("Nothing to clean. Exiting.")
        sys.exit(0)   # Not an error — directory simply has no files to clean

    # Calculate the age cutoff as a Unix timestamp.
    # Files whose modification time is earlier than this will be removed.
    cutoff_time = time.time() - (MAX_AGE_DAYS * 86400)
    cutoff_str  = time.strftime("%Y-%m-%d %H:%M", time.localtime(cutoff_time))
    logger.info(f"Cutoff timestamp: {cutoff_str}")

    removed = 0
    skipped = 0
    errors  = 0

    for filename in sorted(os.listdir(TARGET_DIR)):
        filepath = os.path.join(TARGET_DIR, filename)

        # Only process files, not subdirectories
        if not os.path.isfile(filepath):
            skipped += 1
            continue

        try:
            mtime = os.path.getmtime(filepath)
            if mtime < cutoff_time:
                os.remove(filepath)
                age_days = (time.time() - mtime) / 86400
                logger.info(f"  Removed: {filename} (age: {age_days:.1f} days)")
                removed += 1
            else:
                # File is recent enough to keep — log at debug level
                # so it doesn't clutter the output on busy directories
                logger.debug(f"  Keeping: {filename}")

        except OSError as e:
            logger.error(f"  Error removing {filename}: {e}")
            errors += 1

    # Summary log — always written so there's a clear record of each run
    logger.info(
        f"Cleanup complete: {removed} removed, {skipped} non-file entries skipped, "
        f"{errors} error(s)"
    )

    # Exit with non-zero code if any files could not be removed.
    # This causes systemd to mark the service run as 'failed', which
    # shows up clearly in 'systemctl status' and 'systemctl list-timers'.
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
