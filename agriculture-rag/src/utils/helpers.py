import os
import tempfile
from typing import Union
from PIL import Image
import fitz  # PyMuPDF
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_uploaded_file(uploaded_file, save_dir: str = "data/uploads") -> str:
    """Save uploaded file to disk and return path"""
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def is_pdf(file_path: str) -> bool:
    """Check if file is PDF"""
    return file_path.lower().endswith('.pdf')

def extract_first_page_as_image(pdf_path: str) -> Union[Image.Image, None]:
    """Extract first page of PDF as PIL Image"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        logger.error(f"Error extracting PDF page as image: {str(e)}")
        return None

def create_temp_file(content: bytes, extension: str = "") -> str:
    """Create temporary file with content"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp.write(content)
            return tmp.name
    except Exception as e:
        logger.error(f"Error creating temp file: {str(e)}")
        raise