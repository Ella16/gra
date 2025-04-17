import logging
import logging.config
import os

import yaml

logger = logging.getLogger(__name__)


def set_logger(config_path: str = "logging.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

            stream_log_level = config["handlers"]["console"]["level"]
            file_log_level = config["handlers"]["logfile"]["level"]

            logger.info(
                f"[logger_setting] Stream: {stream_log_level}, File: {file_log_level}"
            )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )
