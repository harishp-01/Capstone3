from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger
import re

logger = get_logger(__name__)

class TextProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Split text into chunks with metadata"""
        cleaned_text = self.clean_text(text)
        chunks = []
        text_chunks = self.text_splitter.split_text(cleaned_text)
        
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "page_num": page_num,
                    "chunk_num": i,
                    "source": "pdf",
                    "type": "text"
                }
            })
        
        logger.debug(f"Split page {page_num} into {len(chunks)} chunks")
        return chunks
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extract important phrases from text (placeholder implementation)"""
        # In a real implementation, you might use NLP techniques here
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        return sorted(word_counts, key=word_counts.get, reverse=True)[:top_n]