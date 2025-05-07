import pytest
from src.embeddings.text_embeddings import TextEmbedder
from src.embeddings.image_embeddings import ImageEmbedder
import numpy as np
from PIL import Image

@pytest.fixture
def text_embedder():
    return TextEmbedder()

@pytest.fixture
def image_embedder():
    return ImageEmbedder()

class TestTextEmbedder:
    def test_embed_text(self, text_embedder):
        embedding = text_embedder.embed_text("sample text")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == text_embedder.get_dimension()

    def test_embed_documents(self, text_embedder):
        docs = [{"text": "doc 1"}, {"text": "doc 2"}]
        result = text_embedder.embed_documents(docs)
        assert len(result) == 2
        assert all("embedding" in doc for doc in result)

class TestImageEmbedder:
    def test_embed_image(self, image_embedder):
        from PIL import Image
        import numpy as np
        
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype='uint8'))
        embedding = image_embedder.embed_image(img)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == image_embedder.get_dimension()