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
