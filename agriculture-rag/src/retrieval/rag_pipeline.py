from typing import Dict, List, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self, vector_store, llm=None):
        self.vector_store = vector_store
        self.llm = llm or self._initialize_llm()
        self.prompt = self._create_prompt()
    
    def _initialize_llm(self):
        """Initialize default LLM"""
        from config import Config
        return OpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_base=Config.LLM_BASE_URL,
            openai_api_key=Config.LLM_API_KEY
        )
    
    def _create_prompt(self) -> PromptTemplate:
        """Create custom prompt for RAG"""
        template = """You are an agriculture expert analyzing documents. Use the following context to answer the question.
        If you don't know the answer, say you don't know. Don't make up answers.

        Context: {context}

        Question: {question}

        Provide a detailed answer based on the context:"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def retrieve_documents(self, query: str, text_embedder, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            query_embedding = text_embedder.embed_text(query)
            results = self.vector_store.search_texts(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, text_embedder) -> Dict:
        """Generate response using RAG pipeline"""
        try:
            # Retrieve relevant documents
            retrieved = self.retrieve_documents(query, text_embedder)
            context = "\n\n".join([
                f"Document {i+1} (Page {doc['document']['metadata']['page_num']+1}):\n{doc['document']['text']}"
                for i, doc in enumerate(retrieved)
            ])
            
            if not context:
                return {
                    "answer": "No relevant information found in documents.",
                    "source_documents": [],
                    "context": ""
                }
            
            # Generate response
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=None,  # We handle retrieval separately
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            
            result = chain({"query": query, "context": context})
            
            return {
                "answer": result["result"],
                "source_documents": [doc["document"] for doc in retrieved],
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error processing your request.",
                "source_documents": [],
                "context": ""
            }
    
    def search_images(self, query: str, image_embedder, k: int = 3) -> List[Dict]:
        """Search for relevant images using text query"""
        try:
            query_embedding = image_embedder.embed_text(query)
            results = self.vector_store.search_images(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Error searching images: {str(e)}")
            return []