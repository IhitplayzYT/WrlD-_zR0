try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF
from tabulate import tabulate
import polars as pl
import re

def extract_pdf(pdf_path: str,describe_image) -> str:
    """
    Extracts:
      - normal text
      - tables (as Markdown)
      - images -> describe_image(image_bytes)
    Returns one large string.
    """
    doc = fitz.open(pdf_path)
    output = []
    for page_num, page in enumerate(doc, start=1):
        output.append(f"\n# Page {page_num}\n")
        text = page.get_text().strip()
        if text:
            output.append(text)
            output.append("\n")
        try:
            tables = page.find_tables()
            for idx, table in enumerate(tables.tables, start=1):
                data = table.extract()
                if data:
                    headers = data[0]
                    rows = data[1:]
                    output.append(f"\n## Table {idx}\n")
                    output.append(tabulate(rows,headers=headers,tablefmt="github"))
                    output.append("\n")
        except Exception:
            pass
        images = page.get_images(full=True)
        for image_idx, image in enumerate(images, start=1):
            xref = image[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            description = describe_image(image_bytes)
            output.append(f"\n[Image {image_idx}]\n"f"{description}\n")
    return "\n".join(output)

def ingest_pdf(pdf_path: str) -> pl.DataFrame:
    """ Extracts numeric data from tables and returns DataFrame with 'price' column """
    doc = fitz.open(pdf_path)
    prices = []
    for page in doc:
        try:
            tables = page.find_tables()
            for table in tables.tables:
                data = table.extract()
                if data and len(data) > 1:
                    for row in data[1:]:
                        for cell in row:
                            # Remove currency symbols, commas, etc.
                            cleaned = re.sub(r'[^\d.]', '', str(cell))
                            if cleaned and cleaned.count('.') <= 1:
                                try:
                                    price = float(cleaned)
                                    if price > 0:  # Only positive prices
                                        prices.append(price)
                                except ValueError:
                                    continue
        except Exception:
            pass
    doc.close()
    if not prices:
        raise ValueError("No numeric price data found in PDF tables")
    return pl.DataFrame({"price": prices})