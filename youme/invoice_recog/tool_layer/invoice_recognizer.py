import json
import re
from copy import deepcopy

import invoice_recog.tool_layer.prompt_for_invoice_recognizer as prompt_iv
from invoice_recog.utils.llm_utils import run_machine

from . import logger

class InvoiceRecognizer(object):
    def __init__(self, tokenizer) -> None:
        self.text = ""
        self.jpeg_files = []
        self.response = {
            "invoice_id": "",
            "total_amount": 0,
            "currency": "",
        }
        self.num_input_tokens, self.num_output_tokens = [], []
        self.tokenizer = tokenizer
        logger.debug(f"[invoice_recognizer] Initialized")

    def reset(self) -> None:
        self.text = ""
        self.jpeg_files = []
        self.response = {
            "invoice_id": "",
            "total_amount": 0,
            "currency": "",
        }
        self.num_input_tokens, self.num_output_tokens = [], []

    def _extract_invoice_info(
        self,
        client,
        model,
        message: list,
        response_fmt: str = "json_object",
    ) -> dict:
        try:
            answer = run_machine(client, model, message, response_fmt)
            self.num_output_tokens.append(self.tokenizer.get_num_tokens(answer))
            response = json.loads(answer)

        except Exception as e:
            logger.error(f"[invoice_recognizer] {e}")
            try:
                answer = run_machine(client, model, message, response_fmt)
                self.num_output_tokens.append(self.tokenizer.get_num_tokens(answer))
                response = json.loads(answer)
            except:
                response = {
                    "invoice_id": "",
                    "billed_amount": 0.0,
                    "currency": "",
                }

        response["invoice_id"] = response["invoice_id"].replace(" ", "").strip()
        response["total_amount"] = response["billed_amount"]
        response.pop("billed_amount", None)

        if type(response["total_amount"]) == str:
            try:
                response["total_amount"] = float(
                    response["total_amount"].replace(",", "")
                )
            except:
                response["total_amount"] = "-"

        if response["currency"] == "YEN":
            response["currency"] = "JPY"

        return response

    def extract_invoice_information(self, client, model, key: str, value: dict) -> dict:
        message = [
            deepcopy(prompt_iv.system_template_IV),
            deepcopy(prompt_iv.human_template_IV),
        ]
        message[1]["content"][0]["text"] = "text_contents: " + value["contents"]
        self.num_input_tokens.append(
            self.tokenizer.get_num_tokens(
                message[0]["content"][0]["text"] + message[1]["content"][0]["text"]
            )
        )

        logger.debug(f"[invoice_recognizer] Extracting invoice information ...")
        response = self._extract_invoice_info(
            client,
            model,
            message=message,
            response_fmt="json_object",
        )
        value["llm_response"] = response
        value["pdf_file_name"] = key

        return value
