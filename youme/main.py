import argparse
import logging
import time

from invoice_recog.config import Config
from invoice_recog.tool_layer_manager import ToolLayerManager
from invoice_recog.utils.logger_setting import set_logger

# Log Settings
set_logger()
logger = logging.getLogger("root")

# Adjust logging level for openai
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


if __name__ == "__main__":
    logger.info(f'{" "*5}{"#"*40}')
    logger.info(f'{" "*10}welcome to YouMe_Us!')
    logger.info(f'{" "*5}{"#"*40}')

    parser = argparse.ArgumentParser(description="YouMe Us - Invoice Recognition")
    parser.add_argument("--exp2", action="store_true", help="Run experiment")
    parser.add_argument("--run", action="store_true", help="Run inference")

    args = parser.parse_args()

    _config = Config(config_path="config.yaml")
    config = _config.get()

    logger.info(f"[main] config: {_config.get_for_print()}")

    tool_layer_manager = ToolLayerManager(config)

    # process flow
    if args.exp2:
        s = time.time()
        num_test_files = tool_layer_manager.experiment_hm()
        logger.critical(
            f"[main] experiment ends in {time.time()-s:.2f} seconds for {num_test_files} files"
        )
        logger.info(f"[main] experiment done")
    elif args.run:
        folder_path = "/data/test/OPP,ODP(테스트)"
        logger.info(f"[main] run start")
        tool_layer_manager.run(folder_path=folder_path)
        logger.info(f"[main] run done")

    logger.info(f'{" "*5}{"#"*40}')
    logger.info(f'{" "*10}see you again @YouMe_Us!')
    logger.info(f'{" "*5}{"#"*40}')
