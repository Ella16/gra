import os
from dataclasses import is_dataclass
from typing import Any, Dict, Optional

import yaml

from . import logger


class Config:
    def __init__(self, config_path: str = ""):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.set_config_from_env()

    """env 변수 중 config dict 에 추가하고 싶은 변수가 있을 시 추가."""

    def set_config_from_env(self):
        # tool_layer/invoice_recognizer
        new_dict_openai = {}
        new_dict_openai["openai_api_key"] = get_from_dict_or_env(
            self.config["tool_layer"]["invoice_recognizer"],
            "openai_api_key",
            "OPENAI_API_KEY",
        )
        new_dict_openai["openai_model_name"] = get_from_dict_or_env(
            self.config["tool_layer"]["invoice_recognizer"],
            "openai_model_name",
            "OPENAI_MODEL_NAME",
        )
        self.config["tool_layer"]["general"].update(new_dict_openai)

        new_dict_output = {}
        new_dict_output["target_keys"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "target_keys",
            "TARGET_KEYS",
        )
        new_dict_output["key_mapping"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "key_mapping",
            "KEY_MAPPING",
        )
        new_dict_output["xlsx_sheet_name"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "xlsx_sheet_name",
            "XLSX_SHEET_NAME",
        )
        new_dict_output["pdf_file_column_name"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "pdf_file_column_name",
            "PDF_FILE_COLUMN_NAME",
        )
        new_dict_output["number_column_name"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "number_column_name",
            "NUMBER_COLUMN_NAME",
        )
        new_dict_output["worker_column_name"] = get_from_dict_or_env(
            self.config["tool_layer"]["output_delivery"],
            "worker_column_name",
            "WORKER_COLUMN_NAME",
        )
        self.config["tool_layer"]["output_delivery"].update(new_dict_output)

        new_dict_serverapi = {}
        new_dict_serverapi["host_port"] = get_from_dict_or_env(
            self.config["server_api"],
            "host_port",
            "HOST_PORT",
        )
        if isinstance(new_dict_serverapi["host_port"], str):
            new_dict_serverapi["host_port"] = int(new_dict_serverapi["host_port"])
        self.config["server_api"].update(new_dict_serverapi)

    def get(self) -> Dict[str, Any]:
        return self.config

    def get_for_print(self) -> Dict[str, Any]:
        """Secret Key 해당하는 내용은 print 시에 제거"""
        from copy import deepcopy

        config_for_print = deepcopy(self.config)
        del config_for_print["tool_layer"]["general"]["openai_api_key"]
        return config_for_print


def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if data:
        if is_dataclass(data) and hasattr(data, key) and getattr(data, key):
            return getattr(data, key)
        elif isinstance(data, Dict) and key in data and data[key]:
            return data[key]
        else:
            return get_from_env(key, env_key, default=default)
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )
