import pytest
from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_pipeline import RAGPipeline
import numpy as np

@pytest.fixture
def vector_store():
    store = VectorStore()
    store.initialize_indexes()
    return store

@pytest.fixture
def rag_pipeline(vector_store):
    return RAGPipeline(vector_store)

class TestVectorStore:
    def test_add_and_search_texts(self, vector_store):
        doc = {
            "text": "sample text",
            "embedding": np.random.rand(384).astype('float32'),
            "metadata": {}
        }
        vector_store.add_texts([doc])
        results = vector_store.search_texts(doc["embedding"], k=1)
        assert len(results) == 1
        assert results[0]["document"]["text"] == doc["text"]

class TestRAGPipeline:
    def test_generate_response(self, rag_pipeline):
        # Mock vector store with sample document
        doc = {
            "text": "Agriculture is the practice of cultivating plants and livestock.",
            "embedding": np.random.rand(384).astype('float32'),
            "metadata": {"page_num": 1}
        }
        rag_pipeline.vector_store.add_texts([doc])
        
        response = rag_pipeline.generate_response(
            "What is agriculture?",
            TextEmbedder()  # Would need to mock this in real test
        )
        assert "answer" in response
        assert "source_documents" in response