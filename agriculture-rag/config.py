import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Application Configuration
    APP_NAME = "Agriculture Document Analysis"
    APP_DESCRIPTION = "AI-powered analysis of agricultural documents using RAG"
    
    # LLM Configuration
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.1
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    
    # Embedding Models
    TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
    
    # Vector Store
    VECTOR_STORE_PATH = "data/vector_store"
    
    # Document Processing
    MAX_PAGE_LENGTH = 1000  # Characters per chunk
    OVERLAP = 200          # Overlap between chunks
    MAX_IMAGE_SIZE = 512   # Max dimension for image processing
    
    # Cache and Storage
    CACHE_DIR = ".cache"
    UPLOAD_DIR = "data/uploads"
    LOG_DIR = "logs"
    
    @staticmethod
    def setup():
        """Initialize required directories"""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)