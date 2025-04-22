import base64
import json
import os
import time
from glob import glob

from . import logger


def flatten_dict(d: dict, desired_keys: list[str] = []) -> dict:
    # 그냥 flatten 시키는 함수. 키 겹치지 않는다고 가정하고, 그냥 아래쪽 키로 덮어 씌움
    # 2중 까지만 있다 친다.
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(v)
        else:
            items[k] = v

    if desired_keys:
        items = {k: items[k] for k in desired_keys}

    return items


# load *.{ext} files from {file_path}
def load_files(file_path: str, ext: str = ".pdf") -> list[str]:
    try:
        file_list = glob(
            file_path if not file_path.endswith("/") else file_path + "*.*"
        )
        num_all_files = len(file_list)
        file_list = [file for file in file_list if file.lower().endswith(ext)]
        file_list.sort()

        logger.info(f"[utils] {len(file_list)} {ext} files loaded,")
        logger.info(f"[utils] among {num_all_files} files from {file_path}")

        return file_list

    except Exception as e:
        logger.error(f"[utils] Error loading files from {file_path}: {e}")
        return []


def setup_experiment(gts_path: str) -> dict[str, any]:
    with open(gts_path, "r", encoding="utf-8") as f:
        gts = json.load(f)
    logger.info(f"[utils] Load {len(gts)} ground truth from {gts_path}")

    new_gts = {}
    for k, v in gts.items():
        if "invoice_no" in v.keys():
            v["invoice_id"] = v.pop("invoice_no")
        if v["currency"] == "YEN":
            v["currency"] = "JPY"
        new_gts[k] = v

    return new_gts


def convert_image_to_base64(image_path: str) -> any:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_current_time() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def get_current_time_for_xlsx() -> str:
    return time.strftime("%m/%d/%Y %H:%M:%S")


def get_current_time_for_xlsxfile() -> str:
    return time.strftime("%Y%m%d")


## METRIC
def compare_gt_answer(gts: dict, llm_result: dict) -> list[str]:
    # Compare the results
    correct = {"all": 0}  # Initialize a dictionary to keep track of correct counts
    test_cases = 0
    not_passed = []

    for key, llm_response in llm_result.items():
        correct_key = {}
        if key not in gts:
            logger.critical(f"\n** Key {key} not found in ground truth **")
            not_passed.append(key)
            continue
        else:
            gt = gts[key]

        test_cases += 1
        check_all = True
        for k, v in gt.items():
            if k in llm_response:
                if llm_response[k] != gt[k]:
                    logger.debug(
                        f"Discrepancy found for {key}: {k} - LLM: {llm_response[k]}, GT: {gt[k]}"
                    )
                    check_all = False
                    correct_key[k] = 0
                else:
                    correct_key[k] = 1
                    if k in correct:
                        correct[k] += 1
                    else:
                        correct[k] = 1

        correct_key["all"] = check_all
        if check_all:
            correct["all"] += 1

        llm_response["compared_gt"] = correct_key

    correct_ratio = {}
    for k, v in correct.items():
        correct_ratio[k] = f"{v / test_cases * 100:.2f}%"

    logger.critical(
        f"Test - Passed : {test_cases} - {correct},\nAccuracy : {correct_ratio}"
    )

    return not_passed


## num tokens
def get_num_tokens_from_string(
    string: str = "", encoding_name: str = "cl100k_base"
) -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    if type(string) != str:
        string = str(string)

    num_tokens = len(encoding.encode(string))
    return num_tokens


## calculate llm api call cost
def calculate_api_call_cost(
    model: str, image_tokens: int, input_tokens: int, output_tokens: int
) -> tuple[float, float, float]:
    pricing = {
        "gpt-4o": {"image": 0.001913, "input": 2.5, "output": 10},
        "gpt-4o-mini": {"image": 0.003825, "input": 0.15, "output": 0.6},
    }  # pricing per 1M tokens

    if model not in pricing:
        model = "gpt-4o"

    image_cost = image_tokens * pricing[model]["image"]
    input_token_cost = input_tokens * pricing[model]["input"] / 1000000
    output_token_cost = output_tokens * pricing[model]["output"] / 1000000

    return (image_cost, input_token_cost, output_token_cost)
