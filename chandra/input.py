from typing import List
import filetype
from PIL import Image
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from chandra.settings import settings


def flatten(page, flag=pdfium_c.FLAT_NORMALDISPLAY):
    rc = pdfium_c.FPDFPage_Flatten(page, flag)
    if rc == pdfium_c.FLATTEN_FAIL:
        print(f"Failed to flatten annotations / form fields on page {page}.")


def load_image(filepath: str):
    image = Image.open(filepath).convert("RGB")
    if image.width < settings.MIN_IMAGE_DIM or image.height < settings.MIN_IMAGE_DIM:
        scale = settings.MIN_IMAGE_DIM / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_pdf_images(filepath: str, page_range: List[int]):
    doc = pdfium.PdfDocument(filepath)
    doc.init_forms()

    images = []
    for page in range(len(doc)):
        if not page_range or page in page_range:
            page_obj = doc[page]
            min_page_dim = min(page_obj.get_width(), page_obj.get_height())
            scale_dpi = (settings.MIN_PDF_IMAGE_DIM / min_page_dim) * 72
            scale_dpi = max(scale_dpi, settings.IMAGE_DPI)
            page_obj = doc[page]
            flatten(page_obj)
            page_obj = doc[page]
            pil_image = page_obj.render(scale=scale_dpi / 72).to_pil().convert("RGB")
            images.append(pil_image)

    doc.close()
    return images


def parse_range_str(range_str: str) -> List[int]:
    range_lst = range_str.split(",")
    page_lst = []
    for i in range_lst:
        if "-" in i:
            start, end = i.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(i))
    page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
    return page_lst


def load_file(filepath: str, config: dict):
    page_range = config.get("page_range")
    if page_range:
        page_range = parse_range_str(page_range)

    input_type = filetype.guess(filepath)
    if input_type and input_type.extension == "pdf":
        images = load_pdf_images(filepath, page_range)
    else:
        images = [load_image(filepath)]
    return images
