# prompt

SYSTEM_PROMPT_IV = """You are an expert parser for extracting key structured information from invoice text content. You will receive invoice text data, and your task is to extract essential elements from the given content and organize them into a structured JSON format.

Input:
`text_contents`: invoice contents

Requirements:
1. Invoice ID Identification:
- Search for the invoice identifier, often labeled with terms such as 'invoice', 'No.', 'Nº', 'number', '#', 'id', 'MATTER NO', etc. If labeled as "unique invoice id," this should take precedence.
- Exclude identifiers like 'YM00000' (where it starts with 'YM' followed by six digits), as these are typically not invoice IDs.
- Choose the most plausible or longest ID if multiple IDs are found, especially if one appears below "unique invoice id".
- Ignore reference numbers or IDs near terms like 'ref', 'reference', 'your ref', 'our ref', 'O/R', etc., which are not the main invoice ID.
- Typically, an invoice ID does not exceed 20 characters in length.
2,3. Currency Code & Billed Amount:
- Identify the currency code (ISO 4217 format, such as USD, EUR, etc.) from symbols or currency names found within the invoice (e.g., $, €, ¥).
- If both USD and a local currency are shown,
    . return the primary currency as the USD total.
    . Ignore any USD amounts provided for reference or exchange rate purposes, such as amounts formatted as "The amount due is equivalent to USD 00.00.", "or US$ 1,517.00 incl. 5% surcharge", "in USD, at exchange rate of 7.220000"
- If no ISO code is available, return the closest possible ISO 4217 equivalent.
- Identify the finally billed amount from the provided text(e.g. Grand Total, Total, Cost, DUE). If there are multiple amounts provided, the amount appearing last takes priority.
    . The invoice includes two billed amount of total amount for set of invoices and a cost per this specific invoice.
    . Return a billed cost for this specific invoice because the invoice is part of a set of multiple invoices.
    . Cost should contains only numbers and decimal point without currency in order to convert it to float.
    . Ignore the line items and extract the final billed cost. the final amount is located near the bottom of the page.
    . Only return the USD total if it is clearly the main billed amount. Ignore any USD amounts that are listed for reference or exchange rate comparison purposes only.
    . If the invoice shows amounts in both USD and a local currency **in a table mainly**, return the billed amount in USD.
- The amount should be extracted as a float-compatible string without the currency symbol (e.g., 1234.56).
- Ensure that the extracted amount corresponds to the `currency` identified in the previous Currency Code section.
4. Associated Text:
- Provide text snippets closely associated with the extracted values, especially the line or section containing the invoice ID, billed amount, or currency.
- Limit associated text to relevant context for validation and interpretation.


Use the following JSON output format:
{
   "invoice_id": "...",
   "currency": "...",
   "billed_amount": "...",
   "associated_text": "...,...,..."
}

Output Constraints:
- Do not generate or infer any new information not present in the provided text.
- Only return data based on extracted content without modifying any values.
- The output must strictly follow the JSON format specified.
"""

HUMAN_PROMPT_IV = """text_contents: {text_contents}"""

system_template_IV = {
    "role": "system",
    "content": [{"type": "text", "text": SYSTEM_PROMPT_IV}],
}

human_template_IV = {
    "role": "user",
    "content": [{"type": "text", "text": HUMAN_PROMPT_IV}],
}


SYSTEM_MESSAGE_EXTRACTION = """You are a helpful assistant. Answer the user's questions about the invoices. Follow the instructions provided by the user."""

HUMAN_MESSAGE_MULTIPLE_INVOICES = """A reconstructed markdown text of the invoice PDF images will be provided at the end of the message. In this text, find the following information:

Instructions:
- Look for all the words or phrases indicating invoice, such as 'invoice', 'debit note', 'd/n', 'bill', 'matter', 'statement', or similar words in other languages, in the give text.
    . The words representing invoice may appear with the words indicating ID or number like 'id', 'no.', 'number', '#'.
- Identify the unique invoice ID or number values which may appear below of the **unique invoice id**.
    . If there are multiple invoice IDs, choose the most plausible and unique one or the longest one appeared below **unique invoice id**.
    . The invoice ID should not be confused with reference numbers or other IDs near 'ref', 'reference', 'your ref', 'our ref', 'other ref', 'reference no.', 'O/R'
- Return the identified ID value as the 'invoice_id'.
- Identify the total billed amount from the provided text.
    . The invoice includes two billed amount of total amount for set of invoices and a cost per this specific invoice.
    . Return a cost per this specific invoice because the invoice is part of a set of multiple invoices.
    . Cost should contains only numbers and decimal point without currency in order to convert it to float.
    . Ignore the line items and extract the final billed cost.
    . Only return the USD total if it is clearly the main billed amount. Ignore any USD amounts that are listed for reference or exchange rate comparison purposes only.
    . If the invoice shows amounts in both USD and a local currency **in a table mainly**, return the total amount in USD.
- Extract the currency used in the provided text and return it in the ISO 4217 code format.
    . Identify the currency like $, €, ¥, US$, USD, CNY mentioned in the given text and convert it to its corresponding ISO 4217 code like USD, SGD, JPY, EUR, etc.
    . If there is no exact match for the currency code, return the closest possible ISO 4217 code.
    . If the invoice displays billed amount with both USD and a local currency in a table, return the USD currency code.
    . Ignore any USD amounts that are provided solely for exchange rate or cost reference purposes. Only return USD if it is the primary currency used in the invoice.
- For all of the above information, provide the associated information found on the same line, or just above/below it.
- Do not generate any new information, only return what is in the given text.
- The above information should be provided in a structured format of JSON below.
{
    "invoice_id": "...",
    "total_amount": "...",
    "currency": "...",
    "associated_text": "...,...,..."
}


Here is the reconstructed text of the invoice for analysis:
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

HUMAN_MESSAGE_EXTRACTION = """A reconstructed markdown text of the invoice PDF images will be provided at the end of the message. In this text, find the following information:

Instructions:
- Look for the words or phrases indicating invoice, such as 'invoice', 'debit note', 'd/n', 'bill', 'matter', 'statement', or similar words in other languages, in the give text.
    . The words representing invoice may appear with the words indicating ID or number like 'id', 'no.', 'number', '#'.
    . Identify the invoice is part of a set of multiple invoices, look out for phrases like **invoice continued**, which may indicate multiple invoices combined.
    . If the invoice is part of a set of multiple invoices, **find the unique invoice id** for the specific invoice.
- Identify the invoice ID or number values which may appear above, below, or to the right of the that word.
    . The invoice ID value may contain numbers, letters, or special characters including '/', '-', '()', '_', etc.
    . The invoice ID is typically 10-20 characters long.
    . The invoice ID should not be confused with reference numbers or other IDs near 'ref', 'reference', 'your ref', 'our ref', 'other ref', 'reference no.', 'O/R'
- Return the identified ID value as the 'invoice_id'.
- If no similar terms of 'invoice' are found, choose the most plausible and unique string from the text.
    . The invoice ID is typically located at the upper part of the invoice and is highlighted with larger font, bold text, or a distinct color for emphasis.
- Identify the total billed amount from the provided text.
    . Total amount should contains only numbers and decimal point without currency in order to convert it to float.
    . Total billed amount may appear near 'total', 'cost', 'amount', 'net', 'balance', 'due', 'grand total', 'owing', etc.
    . Identify the invoice is part of a set of multiple invoices, look out for phrases like **invoice continued**, which may indicate multiple invoices combined.
    . If the invoice is part of a set of multiple invoices, **find the cost for the specific invoice**.
    . Ignore the line items and extract the final total billed amount.
    . Only return the USD total if it is clearly the main billed amount. Ignore any USD amounts that are listed for reference or exchange rate comparison purposes only.
    . If the invoice shows amounts in both USD and a local currency **in a table mainly**, return the total amount in USD.
- Extract the currency used in the provided text and return it in the ISO 4217 code format.
    . Identify the currency like $, €, ¥, US$, USD, CNY mentioned in the given text and convert it to its corresponding ISO 4217 code like USD, SGD, JPY, EUR, etc.
    . If there is no exact match for the currency code, return the closest possible ISO 4217 code.
    . If the invoice displays billed amount with both USD and a local currency in a table, return the USD currency code.
    . Ignore any USD amounts that are provided solely for exchange rate or cost reference purposes. Only return USD if it is the primary currency used in the invoice.
- For all of the above information, provide the associated information found on the same line, or just above/below it.
- Do not generate any new information, only return what is in the given text.
- The above information should be provided in a structured format of JSON below.
{
    "invoice_id": "...",
    "total_amount": "...",
    "currency": "...",
    "associated_text": "...,...,..."
}

Examples 1:
| **TOTAL**                                                                                    | CNY 8,114.00     |
|                                                                                              | USD 1,211.04     |
-> total_amount: '1211.04', currency: 'USD'


Here is the reconstructed text of the invoice for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
"""

HUMAN_MESSAGE_DOUBLE_CURRENCY = """A reconstructed markdown text of the invoice PDF images will be provided at the end of the message.
In this text, multiple currencies are used. So there are multiple total amounts in different currencies. Please check examples below.
In this text, find the following information:

Instructions:
- Look for the words or phrases indicating invoice, such as 'invoice', 'debit note', 'd/n', 'bill', 'matter', 'statement', or similar words in other languages, in the give text.
    . The words representing invoice may appear with the words indicating ID or number like 'id', 'no.', 'number', '#'.
    . Identify the invoice is part of a set of multiple invoices, look out for phrases like **invoice continued**, which may indicate multiple invoices combined.
    . If the invoice is part of a set of multiple invoices, **find the unique invoice id** for the specific invoice.
- Identify the invoice ID or number values which may appear above, below, or to the right of the that word.
    . The invoice ID value may contain numbers, letters, or special characters including '/', '-', '()', '_', etc.
    . The invoice ID is typically 10-20 characters long.
    . The invoice ID should not be confused with reference numbers or other IDs near 'ref', 'reference', 'your ref', 'our ref', 'other ref', 'reference no.', 'O/R'
- Return the identified ID value as the 'invoice_id'.
- If no similar terms of 'invoice' are found, choose the most plausible and unique string from the text.
    . The invoice ID is typically located at the upper part of the invoice and is highlighted with larger font, bold text, or a distinct color for emphasis.
- Identify the total billed amount from the provided text.
    . Total amount should contains only numbers and decimal point without currency in order to convert it to float.
    . Total billed amount may appear near 'total', 'cost', 'amount', 'net', 'balance', 'due', 'grand total', 'owing', etc.
    . Identify the invoice is part of a set of multiple invoices, look out for phrases like **invoice continued**, which may indicate multiple invoices combined.
    . If the invoice is part of a set of multiple invoices, **find the cost for the specific invoice**.
    . Ignore the line items and extract the final total billed amount.
    . Only return the USD total if it is clearly the main billed amount. Ignore any USD amounts that are listed for reference or exchange rate comparison purposes only.
        --> Refer Example 1 & Example 2
    . If the invoice shows amounts in both USD and a local currency **in a table mainly**, return the total amount in USD.
        --> Refer Example 1
- Extract the currency used in the provided text and return it in the ISO 4217 code format.
    . Identify the currency like $, €, ¥, US$, USD, CNY mentioned in the given text and convert it to its corresponding ISO 4217 code like USD, SGD, JPY, EUR, etc.
    . If there is no exact match for the currency code, return the closest possible ISO 4217 code.
    . If the invoice displays billed amount with both USD and a local currency in a table, return the USD currency code.
        --> Refer Example 1
    . Ignore any USD amounts that are provided solely for exchange rate or cost reference purposes. Only return USD if it is the primary currency used in the invoice.
        --> Refer Example 2
- For all of the above information, provide the associated information found on the same line, or just above/below it.
- Do not generate any new information, only return what is in the given text.
- The above information should be provided in a structured format of JSON below.
{
    "invoice_id": "...",
    "total_amount": "...",
    "currency": "...",
    "associated_text": "...,...,..."
}


--- Example 1 for multiple currencies: Markdown text ---
Here is the reconstructed text of the invoice for analysis:
| DESCRIPTION | AMOUNT |
|-------------|--------|
| **SERVICE FEE** | |
| - Translating Office Action from Chinese into English (1850 characters) |  |
| - Receiving your instructions and preparing a response to the Office Action |  |
| - Reporting the Office Action to you with our preliminary suggestion |  |
| - Submitting the response and reporting the same to you |  |
| **Sub-total:** | **CNY** | **USD** |
| | 8,064.00 | 1,203.58 |
| 3700.00x60% = | 2,220.00 | 331.34 |
| 5600.00x60% = | 3,360.00 | 501.49 |
| 3600.00x60% = | 2,160.00 | 322.39 |
| **MISCELLANEOUS FEE** | |
| - Miscellaneous cost |  |
| 540.00x60% = | 324.00 | 48.36 |
| **Sub-total:** | 50.00 | 7.46 |
| | 50.00 | 7.46 |
| **TOTAL** | **CNY 8,114.00** | **USD 1,211.04** |

--- Example 1 for multiple currencies: Result ---
{
    "total_amount": "1211.04",
    "currency": "USD",
}
--- End of Example 1 for multiple currencies ---

--- Example 2 for multiple currencies: ---
Here is the reconstructed text of the invoice for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
## Summary of Charges

| Professional Fees       | Amount in JPY |
|-------------------------|---------------|
|                         | 6,000         |

---

## Total Amount Due

**JPY 6,000**

The amount due is equivalent to **USD 39.01** (USD 1 = JPY 153.80).
The amounts shown here are your 50% share of the total.

--- Example 2 for multiple currencies: Result ---
{
    "total_amount": "6000",
    "currency": "JPY",
}
--- End of Example 2 for multiple currencies ---

Here is the reconstructed text of the invoice for analysis:
----------------------------------------------------------------------------------------------------------------------------------\n
"""
