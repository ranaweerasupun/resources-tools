# © 2025 [Supun Akalanka Sriyananda] — MIT License
"""
app_with_config.py
------------------
A Python service that reads all its configuration from environment
variables — both sensitive secrets (via EnvironmentFile) and
non-sensitive settings (via inline Environment directives).

This pattern separates configuration from code:
  - The application never hardcodes values that might change
  - Secrets live in a restricted file only root can read
  - Rotating a credential means editing one file and restarting

Deploy:
  1. Create the secrets file:
       sudo mkdir -p /etc/env-service
       sudo nano /etc/env-service/config.env     # see config.env.example
       sudo chmod 600 /etc/env-service/config.env
       sudo chown root:root /etc/env-service/config.env

  2. Deploy the service:
       sudo cp env-example.service /etc/systemd/system/
       sudo systemctl daemon-reload
       sudo systemctl enable --now env-example.service
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


def load_and_validate_config():
    """
    Load all required environment variables and fail fast with a
    specific, actionable error message if any are missing.

    Failing fast on startup — rather than crashing deep inside the
    application when a missing value is actually used — produces a
    clear error message in the journal and saves significant debugging time.
    """
    # Define which variables are mandatory. If any of these are absent,
    # the service exits immediately with a non-zero code, which triggers
    # Restart=on-failure. After StartLimitBurst restarts all fail, systemd
    # marks the service as failed — a clear signal to check the config file.
    required_vars = ["API_KEY", "BROKER_HOST"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
        logger.error("Check /etc/env-service/config.env and restart the service.")
        logger.error("Run: sudo systemctl restart env-example.service")
        sys.exit(1)

    return {
        # Secrets — loaded from EnvironmentFile (/etc/env-service/config.env)
        "api_key":     os.environ["API_KEY"],
        "db_password": os.environ.get("DB_PASSWORD"),   # Optional secret

        # Connection config — loaded from EnvironmentFile
        "broker_host": os.environ["BROKER_HOST"],
        "broker_port": int(os.environ.get("BROKER_PORT", "1883")),

        # Behaviour config — set inline in the service file's Environment= directive
        "log_level":   os.environ.get("LOG_LEVEL", "INFO"),
    }


def main():
    config = load_and_validate_config()

    # Confirm loading succeeded — but NEVER log the actual secret values.
    # Logging a secret puts it in the journal where anyone with journalctl
    # access can see it. Log that the key was loaded, not what it says.
    logger.info("Configuration loaded successfully")
    logger.info(f"  Broker:    {config['broker_host']}:{config['broker_port']}")
    logger.info(f"  Log level: {config['log_level']}")
    logger.info(f"  API key:   {'set' if config['api_key'] else 'NOT SET'}")
    logger.info(f"  DB pass:   {'set' if config['db_password'] else 'not set (optional)'}")

    logger.info("Service running. Using loaded configuration for work...")
    counter = 0

    while True:
        counter += 1
        # In a real service, you would use config['broker_host'],
        # config['api_key'], etc. here to do actual work.
        logger.info(f"Work cycle #{counter} — configuration values are available")
        time.sleep(15)


if __name__ == "__main__":
    main()
