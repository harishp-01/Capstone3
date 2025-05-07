import pytest
from src.document_processor.pdf_processor import PDFProcessor
from src.document_processor.text_processor import TextProcessor
from src.document_processor.image_processor import ImageProcessor
import os

@pytest.fixture
def sample_pdf_path():
    return "tests/samples/sample.pdf"

@pytest.fixture
def pdf_processor():
    return PDFProcessor()

@pytest.fixture
def text_processor():
    return TextProcessor()

@pytest.fixture
def image_processor():
    return ImageProcessor()

class TestPDFProcessor:
    def test_process_pdf(self, pdf_processor, sample_pdf_path):
        if not os.path.exists(sample_pdf_path):
            pytest.skip("Sample PDF not found")
            
        text_chunks, images = pdf_processor.process_pdf(sample_pdf_path)
        assert len(text_chunks) > 0
        assert isinstance(images, list)

class TestTextProcessor:
    def test_chunk_text(self, text_processor):
        text = "This is a sample text. " * 100
        chunks = text_processor.chunk_text(text, 0)
        assert len(chunks) > 1
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

class TestImageProcessor:
    def test_process_image(self, image_processor):
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), 'RGB'))
        result = image_processor.process_image(img, 0, 0)
        
        assert "image" in result
        assert "caption" in result
        assert "metadata" in result