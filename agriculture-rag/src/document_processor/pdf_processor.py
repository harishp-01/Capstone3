import fitz  # PyMuPDF
from PIL import Image
import io
import os
from typing import List, Dict, Tuple
import numpy as np
from src.utils.logger import get_logger
from src.document_processor.text_processor import TextProcessor  # Added import
from src.document_processor.image_processor import ImageProcessor  # Explicit import

logger = get_logger(__name__)

class PDFProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()

    def process_pdf(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Process PDF and extract text chunks and images with metadata"""
        text_chunks = []
        images = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    chunks = self.text_processor.chunk_text(text, page_num)
                    text_chunks.extend(chunks)
                
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    image_data = self.image_processor.process_image(image, page_num, img_index)
                    images.append(image_data)
            
            logger.info(f"Processed PDF: {file_path} - {len(text_chunks)} text chunks, {len(images)} images")
            return text_chunks, images
        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise

    def extract_page_as_image(self, pdf_path: str, page_num: int = 0) -> Image.Image:
        """Extract specific page as PIL Image"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(f"Error extracting page {page_num} from {pdf_path}: {str(e)}")
            raise