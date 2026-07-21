from pypdf import PdfReader


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
