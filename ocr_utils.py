import pytesseract
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def ocr_image(file_bytes: bytes) -> str:
    """OCR a single image file (jpg/png)."""
    image = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(image)
    return text

def ocr_pdf(file_bytes: bytes) -> str:
    """OCR a PDF (all pages)."""
    # Convert PDF pages to images
    #images = convert_from_path(io.BytesIO(file_bytes))
    images = convert_from_bytes(file_bytes)
    text_blocks = [pytesseract.image_to_string(img) for img in images]
    return "\n".join(text_blocks)
