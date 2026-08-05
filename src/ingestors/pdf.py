import fitz  # PyMuPDF
from tabulate import tabulate

def extract_pdf(pdf_path: str,describe_image,image_format: str = "png",) -> str:
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