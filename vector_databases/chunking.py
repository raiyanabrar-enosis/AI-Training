from pypdf import PdfReader
import fitz  # pip install pymupdf
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_pdf_pages(path):
    """
    Read a PDF into a list of (page_number, text) records.
    Page numbers are 1-based, the way a human counts pages.
    """
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()
        if extracted and extracted.strip():
            pages.append(
                {
                    "page": i,
                    "text": extracted,
                }
            )

    return pages


def read_pdf_pages_unicode(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        cleaned_text = clean_bengali(text)
        if cleaned_text and cleaned_text.strip():
            pages.append({"page": i, "text": cleaned_text})
    return pages


def clean_bengali(text):
    # Rejoin a word broken by a line break: a Bengali letter, then a
    # newline/space, then a vowel sign or halant that can't start a word.
    text = re.sub(
        r"([\u0980-\u09ff])\s*\n\s*([\u09be-\u09cc\u09cd\u0982\u0983])", r"\1\2", text
    )
    # Collapse doubled vowel signs / nukta / anusvara (আা → আ, ংং → ং)
    text = re.sub(r"([\u09be-\u09cc\u0982\u0983\u09bc])\1+", r"\1", text)
    # Rejoin a word split by newline OR space, when the next piece
    # starts with a vowel sign / halant (can't legally start a word)
    text = re.sub(r'([\u0980-\u09ff])\s+([\u09be-\u09cc\u09cd\u0982\u0983])',
                r'\1\2', text)
    text = re.sub(r'\u09cd[\s\u200c\u200d\u00a0]+', '\u09cd', text)
    # Normalize remaining whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Turn page records into overlapping chunks.

    chunk_size and overlap are measured in CHARACTERS, but we only ever
    cut on spaces, so words stay intact. Each chunk remembers its page.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        length_function=len,
        separators=['\n\n', '\n', '.', '?', '!', " ", ''],
        chunk_overlap=overlap
    )

    for page in pages:
        chunks.extend(splitter.split_text(page["text"]))

    return chunks