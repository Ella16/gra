from invoice_recog.tool_layer.invoice_recognizer import InvoiceRecognizer
from invoice_recog.tool_layer.pdf_reader import PDFReader
from invoice_recog.utils import utils

from . import logger


class FileProcessor(object):
    def __init__(self, tokenizer) -> None:
        self.min_text_line = 10
        self.pdf_reader = PDFReader(tokenizer)
        self.invoice_recognizer = InvoiceRecognizer(tokenizer)
        self.num_pixels = []
        self.num_input_tokens, self.num_output_tokens = [], []
        logger.debug(f"[file_processor] Initialized")

    def _generate_markdown(self, file, client, model) -> tuple[str, dict]:
        # pdf to png : return {fid}.PDF
        fid = self.pdf_reader.convert_pdf_to_img(file)
        # pdf to text
        self.pdf_reader.read_pdf(file)

        markdown_text = self.pdf_reader.build_markdown2(
            client, model, self.min_text_line
            # client, model, self.config["tool_layer"]["general"]["minimum_text_lines"]
        )

        return fid, {
            "contents": markdown_text,
        }

    def _extract_information(
        self, key: str, result: dict, client, model
    ) -> dict[str, any]:
        response = self.invoice_recognizer.extract_invoice_information(
            client, model, key, result
        )
        return response

    def process_file(self, file: str, client, model) -> tuple[dict, tuple[float]]:
        # 검색 시작 시간
        start_timestamp = utils.get_current_time_for_xlsx()

        # generate markdown
        key, result = self._generate_markdown(file, client, model)

        # extract information from markdown
        response = self._extract_information(key, result, client, model)

        self.num_pixels.extend(self.pdf_reader.num_pixels)
        self.num_input_tokens.extend(self.pdf_reader.num_input_tokens)
        self.num_input_tokens.extend(self.invoice_recognizer.num_input_tokens)
        self.num_output_tokens.extend(self.pdf_reader.num_output_tokens)
        self.num_output_tokens.extend(self.invoice_recognizer.num_output_tokens)

        cost = utils.calculate_api_call_cost(
            model,
            len(self.num_pixels),
            sum(self.num_input_tokens),
            sum(self.num_output_tokens),
        )

        # remove temporary files
        self.pdf_reader.reset()
        self.invoice_recognizer.reset()

        response["start_timestamp"] = start_timestamp

        return key, response, cost
