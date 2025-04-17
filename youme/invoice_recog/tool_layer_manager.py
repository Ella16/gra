import concurrent.futures
import json
import os
import time

import openai
import tiktoken
from tqdm import tqdm

from invoice_recog.file_processor import FileProcessor
from invoice_recog.tool_layer.xlsx_handler import xlsxHandler
from invoice_recog.utils import utils

from . import logger


class Tokenizer(object):
    def __init__(self) -> None:
        tiktoken_cache_dir = "./data/tiktoken_cache"
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
        assert os.path.exists(
            os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")
        )
        self.model = tiktoken.get_encoding("cl100k_base")

        cache_files = os.listdir(tiktoken_cache_dir)
        print(f"Files in cache: {cache_files}")

        logger.info(f"[tool_layer_manager] Load tokenizer from {tiktoken_cache_dir}")
        logger.info(f"[tool_layer_manager] Tokenizer initialized")
        logger.info(
            f"[tool_layer_manager] tokens: {self.model.encode('hello youme invoice recognition')} for **hello youme invoice recognition**"
        )
        logger.info(
            f"[tool_layer_manager] Number of tokens: {self.get_num_tokens('hello youme invoice recognition')} for **hello youme invoice recognition**"
        )

    def get_num_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        return len(self.model.encode(text))


class ToolLayerManager(object):
    def __init__(self, config: dict[str, any]) -> None:
        self.config = config
        logger.info(f"[tool_layer_manager] Initialized")
        self.client = openai.OpenAI(
            api_key=self.config["tool_layer"]["general"]["openai_api_key"]
        )
        self.model = self.config["tool_layer"]["general"]["openai_model_name"]
        self.temperature = self.config["tool_layer"]["general"]["openai_temperature"]
        self.xlsx_handler = xlsxHandler(config)
        self.tokenizer = Tokenizer()

    def _run_file(self, file: str, client, model) -> tuple[str, dict]:
        file_processor = FileProcessor(self.config, self.tokenizer)
        key, value, cost = file_processor.process_file(file, client, model)
        return key, value, cost

    def run(self, folder_path: str = "") -> bool:
        # folder path 에 있는 xlsx, PDF 파일 체크
        if not folder_path.endswith("/"):
            folder_path += "/"

        # xlsx handler setting
        pdf_files, _ = self.xlsx_handler.set_target_files(folder_path)

        # run invoice-recognition
        result = {}

        # pretty-print llm result in json
        desired_key_for_pp = [
            "pdf_file_name",
            "invoice_id",
            "currency",
            "total_amount",
            "contents",
        ]

        # 할게 있으면, 달림. 없으면, 끝
        if pdf_files:
            # run parallely
            start = time.time()
            logger.info(f"Run experiment in parallel ... ")
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # executor._max_workers = 5
                    logger.info(f"Number of workers: {executor._max_workers}")
                    logger.info(f"Logical CPU cores: {os.cpu_count()}")

                    logger.info(f"Start invoice recognition ... ")

                    futures = {
                        executor.submit(
                            self._run_file, file, self.client, self.model
                        ): file
                        for file in pdf_files
                    }
                    for future in tqdm(concurrent.futures.as_completed(futures)):
                        pdf_file, value, cost = future.result()
                        value["cost"] = cost
                        value["end_timestamp"] = utils.get_current_time_for_xlsx()
                        result[pdf_file] = value

                        # write results to txt files
                        value_for_pp = utils.flatten_dict(value, desired_key_for_pp)
                        value_for_pp["contents"] = (
                            value_for_pp["contents"]
                            .replace("\n```", "")
                            .replace("```markdown\n", "")
                        )

                        pp_path: str = (
                            f"{folder_path}{value_for_pp['pdf_file_name']}.JSON"
                        )
                        with open(
                            pp_path,
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(value_for_pp, f, indent=4, ensure_ascii=False)
                        value_for_pp.clear()

            except Exception as e:
                logger.error(f"[tool_layer_manager] Error in parallel processing: {e}")
                return False

            # cost check
            total_cost = sum([sum(v["cost"]) for v in result.values()])
            logger.info(
                f"*** Cost: {total_cost:.2f} USD for {len(result.keys())} files ***"
            )
            logger.info(
                f"*** Time: {time.time()-start:.2f} seconds for {len(pdf_files)} ***"
            )

            # convert result to xlsx data format
            try:
                self.xlsx_handler.run(result)
            except Exception as e:
                logger.error(f"[tool_layer_manager] Error in xlsx handling: {e}")
                return False

            return True

        else:
            logger.info(f"[tool_layer_manager] No PDF files found in {folder_path}")
            logger.info(f"[tool_layer_manager] Invoice recognition done")
            return False
