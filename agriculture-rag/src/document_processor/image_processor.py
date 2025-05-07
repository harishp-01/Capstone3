from PIL import Image
import numpy as np
from typing import Dict
from src.utils.logger import get_logger
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

logger = get_logger(__name__)

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing BLIP model on {self.device}")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
    
    def process_image(self, image: Image.Image, page_num: int, img_index: int) -> Dict:
        """Process image and generate caption with metadata"""
        try:
            # Generate caption
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            return {
                "image": image,
                "caption": caption,
                "metadata": {
                    "page_num": page_num,
                    "img_index": img_index,
                    "source": "pdf",
                    "type": "image"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.LANCZOS)
        return image