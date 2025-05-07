from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict
import torch
import numpy as np
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading image embedding model: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dimension = self.model.config.projection_dim
    
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed single image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error embedding image: {str(e)}")
            raise
    
    def embed_images(self, images: List[Dict]) -> List[Dict]:
        """Embed list of images"""
        if not images:
            return []
            
        try:
            for img_data in images:
                embedding = self.embed_image(img_data["image"])
                img_data["embedding"] = embedding
            
            logger.info(f"Embedded {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error embedding images: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension