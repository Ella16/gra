# ocr free pdf processor
import sys
sys.path.append('./youme')
from invoice_recog.file_processor import FileProcessor
from invoice_recog.tool_layer_manager import Tokenizer
import keys
from openai import OpenAI

openai_client = OpenAI(api_key=keys.openai.api_key, organization=keys.openai.organization)
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_ENGINE = "text-embedding-ada-002"
model_name =  GPT_MODEL
temperature = 0.7

class OCRFreeFileProcessor(FileProcessor):
    def process_file(self, file: str, client, model) -> tuple[dict, tuple[float]]:
        
        # generate markdown
        key, result = self._generate_markdown(file, client, model)

        # extract information from markdown
        response = self._extract_information(key, result, client, model)

        # self.num_pixels.extend(self.pdf_reader.num_pixels)
        # self.num_input_tokens.extend(self.pdf_reader.num_input_tokens)
        # self.num_input_tokens.extend(self.invoice_recognizer.num_input_tokens)
        # self.num_output_tokens.extend(self.pdf_reader.num_output_tokens)
        # self.num_output_tokens.extend(self.invoice_recognizer.num_output_tokens)

        # cost = utils.calculate_api_call_cost(
        #     model,
        #     len(self.num_pixels),
        #     sum(self.num_input_tokens),
        #     sum(self.num_output_tokens),
        # )
        
        # remove temporary files
        self.pdf_reader.reset()
        self.invoice_recognizer.reset()

        # response["start_timestamp"] = start_timestamp

        return key, response

import os
tokenizer = Tokenizer(tiktoken_cache_dir = "./youme/data/tiktoken_cache")
file_proc = OCRFreeFileProcessor(tokenizer)
data_dir = './data/250415'
pdfs = [os.path.join(data_dir, pdf) for pdf in os.listdir(data_dir) if pdf.endswith('.pdf')]

file = pdfs[0]
key, value, cost = file_proc.process_file(file, openai_client, model_name)
print()