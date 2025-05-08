import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, text_dim: int = 384, image_dim: int = 512):
        self.text_index = None
        self.image_index = None
        self.text_metadata = []
        self.image_metadata = []
        self.text_dim = text_dim
        self.image_dim = image_dim
        
    def initialize_indexes(self):
        """Initialize empty FAISS indexes"""
        self.text_index = faiss.IndexFlatL2(self.text_dim)
        self.image_index = faiss.IndexFlatL2(self.image_dim)
        logger.info("Initialized empty FAISS indexes")
    
    def add_texts(self, documents: List[Dict]):
        """Add text documents to vector store"""
        if not documents:
            return
            
        if self.text_index is None:
            self.initialize_indexes()
        
        try:
            embeddings = np.array([doc["embedding"] for doc in documents]).astype('float32')
            self.text_index.add(embeddings)
            self.text_metadata.extend(documents)
            logger.info(f"Added {len(documents)} text documents to vector store")
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            raise
    
    def add_images(self, images: List[Dict]):
        """Add images to vector store"""
        if not images:
            return
            
        if self.image_index is None:
            self.initialize_indexes()
        
        try:
            embeddings = np.array([img["embedding"] for img in images]).astype('float32')
            self.image_index.add(embeddings)
            self.image_metadata.extend(images)
            logger.info(f"Added {len(images)} images to vector store")
        except Exception as e:
            logger.error(f"Error adding images to vector store: {str(e)}")
            raise
    
    def search_texts(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar text documents"""
        if self.text_index is None or len(self.text_metadata) == 0:
            return []
            
        try:
            distances, indices = self.text_index.search(query_embedding.reshape(1, -1).astype('float32'), k)
            return [
                {
                    "document": self.text_metadata[idx],
                    "score": float(distances[0][i])
                }
                for i, idx in enumerate(indices[0])
                if idx != -1
            ]
        except Exception as e:
            logger.error(f"Error searching texts: {str(e)}")
            return []
    
    def search_images(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar images"""
        if self.image_index is None or len(self.image_metadata) == 0:
            return []
            
        try:
            distances, indices = self.image_index.search(query_embedding.reshape(1, -1).astype('float32'), k)
            return [
                {
                    "image": self.image_metadata[idx],
                    "score": float(distances[0][i])
                }
                for i, idx in enumerate(indices[0])
                if idx != -1
            ]
        except Exception as e:
            logger.error(f"Error searching images: {str(e)}")
            return []
    
    def save(self, base_path: str):
        """Save vector store to disk with path validation"""
        try:
            # Convert property to string if needed
            if hasattr(base_path, '__class__') and isinstance(base_path, property):
                base_path = base_path.fget()
                
            logger.info(f"Saving vector store to: {base_path}")
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            # Save text index and metadata
            if self.text_index is not None:
                faiss.write_index(self.text_index, f"{base_path}_text.faiss")
                with open(f"{base_path}_text_meta.pkl", "wb") as f:
                    pickle.dump(self.text_metadata, f)
            
            # Save image index and metadata
            if self.image_index is not None:
                faiss.write_index(self.image_index, f"{base_path}_image.faiss")
                with open(f"{base_path}_image_meta.pkl", "wb") as f:
                    pickle.dump(self.image_metadata, f)
            
            logger.info(f"Successfully saved vector store to {base_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load(self, base_path: str) -> bool:
        """Load vector store from disk with error handling"""
        try:
            # Clear existing data
            self.text_index = None
            self.image_index = None
            self.text_metadata = []
            self.image_metadata = []
            
            # Load text index if exists
            text_index_path = f"{base_path}_text.faiss"
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
                with open(f"{base_path}_text_meta.pkl", "rb") as f:
                    self.text_metadata = pickle.load(f)
            
            # Load image index if exists
            image_index_path = f"{base_path}_image.faiss"
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
                with open(f"{base_path}_image_meta.pkl", "rb") as f:
                    self.image_metadata = pickle.load(f)
            
            if self.text_index or self.image_index:
                logger.info(f"Loaded vector store from {base_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # If loading fails, reinitialize empty indexes
            self.initialize_indexes()
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "text_documents": len(self.text_metadata),
            "images": len(self.image_metadata),
            "text_index_exists": self.text_index is not None,
            "image_index_exists": self.image_index is not None
        }