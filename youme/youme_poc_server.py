import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import uvicorn
from fastapi import FastAPI

from invoice_recog.config import Config
from invoice_recog.tool_layer_manager import ToolLayerManager
from invoice_recog.utils.logger_setting import set_logger

# Log Settings
set_logger()
logger = logging.getLogger("root")

# Adjust logging level for openai
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Parallel processing
executor = ProcessPoolExecutor(max_workers=1)

# Config setting
_config = Config("config.yaml")
config = _config.get()
logger.info(f"[main] config: {_config.get_for_print()}")

# ToolLayerManager
tool_layer_manager = ToolLayerManager(config)

# uvicorn FastAPI
app = FastAPI()


# Define the filter
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            record.args and len(record.args) >= 3 and record.args[2] != "/healthcheck"
        )


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@app.get("/api/invoice/healthcheck")
async def health_check():
    logger.debug(f"[HEALTH_CHECK] {{'status': 'ok'}}")
    return {"status": "ok"}


def invoice_recognition_run(folder_path: str = "") -> None:
    finished = tool_layer_manager.run(folder_path)
    if finished:
        logger.info(f"[GET_KEY_SEARCH] task finished for path: {folder_path}")
    else:
        logger.info(f"[GET_KEY_SEARCH] task failed for path: {folder_path}")

    logger.info(f"[GET_KEY_SEARCH] waiting for next task...")


@app.get("/api/invoice/getKeySearch")
async def invoice_get_key_search(path: Optional[str] = None):
    if not path:
        return {"status": "No path provided. Please provide a path."}

    logger.info(f"[GET_KEY_SEARCH] task submitted for path: {path}")
    executor.submit(invoice_recognition_run, path)

    return {"status": "Invoice recognition process started."}


if __name__ == "__main__":
    port = config["server_api"]["host_port"]
    host_ip = config["server_api"]["host_ip"]
    host = config["server_api"]["host"]

    uvicorn.run(
        "youme_poc_server:app", host=host_ip, port=port, reload=False
    )  # , reload=True)
