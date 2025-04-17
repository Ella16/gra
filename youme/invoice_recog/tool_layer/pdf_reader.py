import base64
import json
import os
from copy import deepcopy
from io import BytesIO

import PyPDF2  # pypdf2==3.0.1
from pdf2image import convert_from_path  # pdf2image==1.17.0

from invoice_recog.tool_layer import prompt_for_pdf_reader as prompt_pdf
from invoice_recog.utils.llm_utils import run_machine

from . import logger


class PDFReader(object):
    def __init__(self, tokenizer) -> None:
        # self.config = config["tool_layer"]["pdf_reader"]
        # self.target_dpi = (
        #     self.config["target_dpi"] if "target_dpi" in self.config else 144
        # )
        self.target_dpi = 300 
        self.pdf_file, self.markdown_text = "", ""
        self.extracted_text, self.num_text_lines, self.base64_images = [], [], []
        self.num_input_tokens, self.num_pixels, self.num_output_tokens = [], [], []
        self.tokenizer = tokenizer
        logger.debug(f"[pdf_reader] Initialized")

    def reset(self) -> None:
        self.pdf_file, self.markdown_text = "", ""
        self.extracted_text, self.num_text_lines, self.base64_images = [], [], []
        self.num_input_tokens, self.num_pixels, self.num_output_tokens = [], [], []
        logger.debug(f"[pdf_reader] Reset and remove temporary files")

    def convert_pdf_to_img(self, file: str = "") -> str:
        self.pdf_file = file
        fid = file.split("/")[-1]
        base64_images = []
        try:
            images = convert_from_path(pdf_path=file, fmt="png", dpi=self.target_dpi)
            # root_dir = self.config["image_dir"]
            for _, image in enumerate(images):
                # for cost calculation
                self.num_pixels.append(1)
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                base64_images.append(
                    f'data:image/png;base64,{base64.b64encode(image_bytes).decode("utf-8")}'
                )
        except Exception as e:
            logger.error(f"[pdf_reader] Error converting PDF to image: {file}")
            logger.error(f"[pdf_reader] Error converting PDF to image: {e}")

        self.base64_images = base64_images
        return fid

    def _read_pdf(self, pdf_file: str) -> list[tuple[str, int]]:
        try:
            results = []
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text, num_text = "", 0
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    num_text = len(text.split("\n"))
                    results.append((text, num_text))
            return results
        except Exception as e:
            logger.error(f"[pdf_reader] Error reading PDF file **{pdf_file}**: {e}")
            return [("", 0)]

    def read_pdf(self, file: str = "") -> None:
        if not file:
            file = self.pdf_file
        pages = self._read_pdf(file)
        self.extracted_text = [page[0] for page in pages]
        self.num_text_lines = [page[1] for page in pages]
        logger.debug(
            f"[pdf_reader] {sum(self.num_text_lines)} lines extracted from {file}"
        )

    def build_markdown2(self, client, model, minimum_text_lines: int = 10) -> str:
        logger.debug(f"[pdf_reader] Building markdown ... ")
        markdown_text = ""

        for i, image_url in enumerate(self.base64_images):
            pagei, num_text_linei = self.extracted_text[i], self.num_text_lines[i]
            try:
                human_prompt = deepcopy(prompt_pdf.human_template_MD_Extractor)
                if num_text_linei < minimum_text_lines:
                    human_prompt["content"][0][
                        "text"
                    ] = "text_contents: ** No Contents Available for this page **"
                else:
                    human_prompt["content"][0]["text"] = "text_contents: " + pagei
                message = [
                    deepcopy(prompt_pdf.system_template_MD_Extractor),
                    human_prompt,
                ]
                self.num_input_tokens.append(self.tokenizer.get_num_tokens(message))
                message[1]["content"][1]["image_url"]["url"] = image_url

                md = run_machine(client, model, messages=message)
                self.num_output_tokens.append(self.tokenizer.get_num_tokens(md))

                markdown_text += md

            except Exception as e:
                logger.error(f"[pdf_reader] Error building markdown: {e}")

        logger.debug(f"[pdf_reader] Built markdown")

        self.markdown_text = markdown_text
        return markdown_text
