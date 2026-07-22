from pypdf import PdfReader
import fitz  # pip install pymupdf
import re


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

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            # Grow the window word by word until we hit chunk_size chars.
            current = []
            length = 0
            i = start
            while i < len(words) and length < chunk_size:
                current.append(words[i])
                length += len(words[i]) + 1  # +1 for the space
                i += 1

            chunk_text = " ".join(current)
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page["page"],
                }
            )

            if i >= len(words):
                break

            # Step forward, but leave `overlap` characters of context behind.
            step = max(1, len(current) - _words_for(current, overlap))
            start += step

    return chunks


def _words_for(words, target_chars):
    """How many trailing words make up roughly `target_chars` characters."""
    count = 0
    length = 0

    for w in reversed(words):
        length += len(w) + 1
        count += 1
        if length >= target_chars:
            break

    return count
