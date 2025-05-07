from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text string"""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """Embed list of document chunks"""
        if not documents:
            return []
            
        texts = [doc["text"] for doc in documents]
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            
            for i, doc in enumerate(documents):
                doc["embedding"] = embeddings[i]
            
            logger.info(f"Embedded {len(documents)} text chunks")
            return documents
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension