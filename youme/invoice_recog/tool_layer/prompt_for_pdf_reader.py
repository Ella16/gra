SYSTEM_PROMPT_MD_Extractor = """You are an assistant that converts extracted text from PDF documents into structured Markdown format. I will provide either a text input derived from the PDF content or an image URL representing the PDF document. Your task is to analyze the provided content and recreate the document’s layout, headings, lists, tables, and formatting to accurately reflect the original PDF's structure.

Output Requirements:
- Recreate the document layout as closely as possible using Markdown.
- Include all headings, subheadings, bullet points, tables, and formatted text.
- Ensure that any visual structure (such as bold text, sections, or bullet points) is translated into Markdown syntax.
- Maintain the order and structure of the original content with no missing information.
- Generate output in Markdown format only, without any explanations or additional commentary.
- If `text_contents` is missing or appears incorrectly decoded, ignore it and use only the `image` input to create the Markdown output.
- Pay special attention to sections related to Invoice ID, Currency, and Amount, ensuring these details are accurately captured.

Input Information:
- `image`: image URL or base64 image of PDF documents.
- `text_contents`: Text extracted from the PDF document.

Expected Markdown Format:
- Use `#`, `##`, `###` for headings, based on the PDF's hierarchy.
- Use `*` or `-` for bullet points and `1.` for ordered lists.
- Format tables using Markdown table syntax.
- Emphasize text where necessary using `**bold**` or `_italic_` syntax."""

HUMAN_PROMPT_MD_Extractor = """text_contents: {text_contents}"""

system_template_MD_Extractor = {
    "role": "system",
    "content": [{"type": "text", "text": SYSTEM_PROMPT_MD_Extractor}],
}

human_template_MD_Extractor = {
    "role": "user",
    "content": [
        {"type": "text", "text": HUMAN_PROMPT_MD_Extractor},
        {
            "type": "image_url",
            "image_url": {"url": "image_data"},
        },
    ],
}


SYSTEM_MESSAGE_MD = (
    """You are a helpful assistant. Follow the instructions provided by the user."""
)

HUMAN_MESSAGE_MD = """Convert the following image to markdown.
Return only the markdown with no explanation text.
Do not exclude any content from the page.

Images of invoices are as follows:
"""

HUMAN_MESSAGE_REVISE_1 = """I will provide you with the markdown for the image below.
Check the markdown and compare it with the image.
If there are any discrepancies between the markdown and the image, correct the markdown.
If markdown misses any text from the image, please add them to the markdown, and revise the markdown.
Return only the revised markdown with no explanation text.

Images of invoices are as follows:
"""

HUMAN_MESSAGE_REVISE_2 = """\n\n
Markdown converted from images of invoices are as follows:
"""

HUMAN_MESSAGE_REVISE_3 = """A reconstructed markdown text of the invoice PDF images and an extracted text string from these images directly will be provided at the end of the message.
The reconstructed markdown text contains more detailed content, but the extracted text is more accurate.
Please compare the reconstructed markdown text with the extracted text string and follow the instructions below.

Instructions:
- Compare the reconstructed markdown text and the extracted text.
- Identify any discrepancies between the two texts.
    . Ignore differences in line breaks or formatting, but find sections where the spelling is similar but not identical.
- Replace any incorrect information in the reconstructed markdown text with the corresponding correct information from the extracted text, using the extracted text as the reference.
- Return the entire corrected markdown text as 'markdown'
- Return a list of the individual replacements in the form of '(..., ...)' from the reconstructed and from the extracted.
- Provide no additional explanations, just the necessary information.
- Do not generate any new information, only return what is in the given text.
- The above information should be provided in a structured format of JSON below.
{
    "markdown": "...",
    "replacements": "(...,...),(...,...),(...,...)",
}

Example:
Reconstructed text: 'The total amount\nis $1,0000.00.\nInvoice ID: U1234567890'
Extracted text: 'The total amount is $1,000.00.\nInvoice ID: U1234567B00'
-->
{
    "markdown": "The total amount\nis $1,000.00.\nInvoice ID: U1234567B00",
    "replacements": "(1,0000.00, 1,000.00),(7890, 7B00)",
}

Here is the reconstructed text of the invoice image for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
"""

HUMAN_MESSAGE_REVISE_4 = """
Here is the extracted text from the invoice directly for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
"""

HUMAN_MESSAGE_CURRENCY_CHECK = """A reconstructed markdown text of the invoice PDF images will be provided at the end of the message. In this text, find the following information:

Instructions:
- Look for all the words, phrases, symbols indicating currency, such as '$', '€', '¥', 'US$', 'USD', 'CNY', 'US dollars', etc.
    . Without duplications, list up the raw currency symbols and phrases found in the text.
- Identify and list up the currencies used in the provided text and return them in the ISO 4217 code format.
    . Remove duplicated currency codes in the response.
- For all of the above information, provide the associated information found on the same line, or just above/below it.
- Do not generate any new information, only return what is in the given text.
- The above information should be provided in a structured format of JSON below.
{
    "currency_raw": "...,...,...",
    "currency": "...,...,...",
    "associated_text": "...,...,...",
}


Here is the reconstructed text of the invoice for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
"""
