import os
import warnings
from contextlib import suppress
import streamlit as st
from PIL import Image
from config import Config
from src.document_processor.pdf_processor import PDFProcessor
from src.embeddings.text_embeddings import TextEmbedder
from src.embeddings.image_embeddings import ImageEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.helpers import save_uploaded_file, is_pdf, extract_first_page_as_image
from src.utils.logger import get_logger

# Suppress specific PyTorch/Streamlit warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch._classes")

# Initialize configuration and logging
config = Config()
config.setup()
logger = get_logger(__name__)

# Initialize components
with suppress(RuntimeError):  # Suppress the torch._classes error
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    vector_store = VectorStore()

    # Handle vector store loading with error recovery
    if not vector_store.load(config.VECTOR_STORE_PATH):
        logger.warning("Could not load vector store, initializing empty")
        vector_store.initialize_indexes()

    pdf_processor = PDFProcessor()
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        text_embedder=text_embedder
    )
    

# Streamlit UI Configuration
st.set_page_config(
    page_title="Agriculture Document Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for document upload
with st.sidebar:
    st.title("🌾 Agriculture Document Analysis")
    st.markdown("""
    Upload agricultural documents (PDFs or images) and interact with them using AI.
    The system supports:
    - Text extraction and analysis
    - Image understanding
    - Question answering
    - Document search
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload PDF documents or images related to agriculture"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                file_path = save_uploaded_file(uploaded_file)
                
                if is_pdf(file_path):
                    # Process PDF
                    text_chunks, images = pdf_processor.process_pdf(file_path)
                    
                    # Generate embeddings
                    text_chunks = text_embedder.embed_documents(text_chunks)
                    images = image_embedder.embed_images(images)
                    
                    # Add to vector store
                    vector_store.add_texts(text_chunks)
                    vector_store.add_images(images)
                    
                    # Save vector store - FIXED: Call the property to get the string path
                    vector_store.save(Config().VECTOR_STORE_PATH)
                    
                    st.success("PDF processed successfully!")
                    
                    # Show preview
                    preview_image = extract_first_page_as_image(file_path)
                    if preview_image:
                        st.image(
                            preview_image,
                            caption="First page preview",
                            use_container_width=True
                        )
                else:
                    # Process image
                    image = Image.open(file_path)
                    image_data = {
                        "image": image,
                        "caption": "Uploaded image",
                        "metadata": {
                            "page_num": 0,
                            "img_index": 0,
                            "source": "upload",
                            "type": "image"
                        }
                    }
                    
                    # Generate embedding
                    image_data = image_embedder.embed_images([image_data])[0]
                    
                    # Add to vector store
                    vector_store.add_images([image_data])
                    vector_store.save(Config().VECTOR_STORE_PATH)  # FIXED: Call the property
                    
                    st.success("Image processed successfully!")
                    st.image(image, caption="Uploaded image", use_container_width=True)
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                logger.error(f"Document processing error: {str(e)}")

# Main chat interface
st.title("Chat with Agriculture Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about agriculture documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response = rag_pipeline.generate_response(prompt)
                
                st.markdown(response["answer"])
                
                if response["source_documents"]:
                    with st.expander("View source documents"):
                        for doc in response["source_documents"]:
                            st.write(f"**Page {doc.metadata['page_num'] + 1}**")
                            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                            st.divider()
            
            except Exception as e:
                error_msg = "Sorry, I encountered an error processing your request."
                st.error(error_msg)
                logger.error(f"Chat error: {str(e)}")
                response = {"answer": error_msg, "source_documents": []}
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})