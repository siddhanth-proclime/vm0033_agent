import os
import streamlit as st
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pandas as pd
import ast
from streamlit_pdf_viewer import pdf_viewer

# Page configuration
st.set_page_config(
    page_title="VM0033 RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for page number
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

@st.cache_resource
def create_vecdb(filename="embeddings.csv", 
                 persist_directory="chroma_vm0033_db", 
                 collection_name="vm0033_rag"):
    """Create vector database from embeddings CSV"""
    df = pd.read_csv(filename)
    df['metadata'] = df['metadata'].apply(ast.literal_eval)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"},
    )
    
    texts = df["text"].tolist()
    metadatas = df["metadata"].tolist()
    ids = df["id"].astype(str).tolist()
    
    vecdb_chroma = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    
    vecdb_chroma.persist()
    return vecdb_chroma

@st.cache_resource
def initialize_vectordb():
    """Initialize the vector database"""
    persist_directory = "chroma_vm0033_db"
    collection_name = "vm0033_rag"
    
    # Check and create vector DB if needed
    if not os.path.isdir(persist_directory):
        st.info("Vector database not found. Creating it now...")
        create_vecdb()
        st.success("Vector database created successfully!")
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"},
    )
    
    # Load the persisted Chroma DB
    vecdb_chroma = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vecdb_chroma

def initialize_llm(api_key, base_url):
    """Initialize LLM with provided credentials"""
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model="meta-llama/llama-3.3-8b-instruct:free",
        default_headers={}
    )
    return llm

def RAG(query, vecdb_chroma, llm, k=3):
    """Execute RAG query and return results"""
    try:
        # Create QA bot with specified k value
        qabot = RetrievalQA.from_chain_type(
            chain_type="stuff",
            llm=llm,
            retriever=vecdb_chroma.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
        )
        
        # Execute query
        result = qabot.invoke(dict(query=query))
        return {
            "success": True,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Configuration
st.sidebar.subheader("🔑 API Credentials")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    type="password",
    value="",
    help="Enter your OpenRouter API key"
)

openrouter_base_url = st.sidebar.text_input(
    "OpenRouter Base URL",
    value="https://openrouter.ai/api/v1",
    help="Enter the OpenRouter base URL"
)

st.sidebar.markdown("---")

k_value = st.sidebar.slider(
    "Number of source documents to retrieve (k)",
    min_value=1,
    max_value=10,
    value=3,
    help="Select how many relevant documents to retrieve for answering your query"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.info(
    "This RAG system retrieves relevant documents and uses AI to answer your questions. "
    "Adjust the 'k' value to control how many source documents are used."
)

# Initialize the vector database
try:
    with st.spinner("Loading vector database..."):
        vecdb_chroma = initialize_vectordb()
    st.success("Vector database ready!")
except Exception as e:
    st.error(f"Failed to initialize vector database: {str(e)}")
    st.stop()

# Create two columns: main content and PDF viewer
col1, col2 = st.columns([49, 50-1])

with col1:
    # Title and description
    st.title("VM0033 RAG")
    st.markdown("Ask questions about your documents and get AI-powered answers with source references.")
    st.markdown("---")

    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., List monitoring requirements and frequency.",
        help="Type your question here and press Enter"
    )

    # Submit button
    if st.button("🔍 Get Answer", type="primary"):
        if not openrouter_api_key:
            st.warning("⚠️ Please enter your OpenRouter API Key in the sidebar to proceed.")
        elif query:
            try:
                # Initialize LLM with provided credentials
                llm = initialize_llm(openrouter_api_key, openrouter_base_url)
                
                with st.spinner("Searching and generating answer..."):
                    result = RAG(query, vecdb_chroma, llm, k=k_value)
                
                if result["success"]:
                    # Store results in session state
                    st.session_state.answer = result['answer']
                    st.session_state.source_documents = result['source_documents']
                    
                else:
                    # Display error
                    st.error("❌ An error occurred while processing your query.")
                    st.markdown(f"**Error Details:** {result['error']}")
                    st.warning("Please try another query or check your configuration.")
            except Exception as e:
                st.error(f"❌ Error initializing LLM: {str(e)}")
                st.warning("Please check your API credentials and try again.")
        else:
            st.warning("Please enter a question.")
    
    # Display answer if it exists in session state
    if 'answer' in st.session_state:
        st.markdown("### 💡 Answer")
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; color: black;'>{st.session_state.answer}</div>", unsafe_allow_html=True)

    # Example queries
    st.markdown("---")
    st.markdown("### 💭 Example Queries")
    st.markdown("""
    - List monitoring requirements and frequency.
    - What are the applicability conditions?
    - How is the scenario determined?
    """)

with col2:
    st.subheader("📄 PDF Viewer")
    
    # Display source documents as buttons
    if 'source_documents' in st.session_state and st.session_state.source_documents:
        st.markdown("### 📚 Source Documents")
        for idx, doc in enumerate(st.session_state.source_documents, 1):
            page_num = doc.metadata.get('page_number', 1)
            source_name = doc.metadata.get('source', 'VM0033 Methodology')
            
            # Create a button for each source document
            if st.button(f"📄 Source {idx} - Page {page_num}", key=f"doc_{idx}"):
                st.session_state.current_page = page_num
                st.rerun()
        
        st.markdown("---")
    
    pdf_path = "VM0033_Methodology.pdf"
    
    if os.path.exists(pdf_path):
        # st.write(f"Current page: {st.session_state.current_page}")
        
        # Display the PDF using streamlit-pdf-viewer
        pdf_viewer(
            input=pdf_path,
            width="100%",
            height=800,
            zoom_level="auto",
            viewer_align="center",
            show_page_separator=True,
            scroll_to_page=st.session_state.current_page
        )
        
        # Optional: direct link to open PDF in a new tab
        st.markdown(f"[📄 Open full PDF]({pdf_path})", unsafe_allow_html=True)
    else:
        st.error("PDF file not found. Please ensure 'VM0033_Methodology.pdf' is in the same directory.")
