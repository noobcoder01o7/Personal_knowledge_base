import streamlit as st
import os
import time

# --- Corrected LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings  # FIX: Use correct Ollama imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Constants ---
DATA_PATH = "documents"
CHROMA_PATH = "chroma"

# FIX: Use the fast and correct model for embeddings
# This MUST match the model used for ingestion
EMBEDDING_MODEL = "nomic-embed-text" 

# FIX: Use the powerful model for generating answers
LLM_MODEL = "llama3"


st.set_page_config(page_title="AI Personal Knowledge Base", layout="wide")
st.title("🤖 AI-Powered Personal Knowledge Base")
st.markdown(f"Ask questions about the content in your 'documents' folder. (Using `{LLM_MODEL}` for answers and `{EMBEDDING_MODEL}` for search.)")

# --- Ingestion Logic ---
# Check if the vector database exists
if not os.path.exists(CHROMA_PATH):
    st.info("Chroma database not found. Starting ingestion process... (This may take a few minutes the first time)")
    
    with st.spinner("Loading and splitting documents..."):
        # Load PDFs
        pdf_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader, 
            silent_errors=True
        )
        # Load TXTs
        txt_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt",
            silent_errors=True
        )
        
        pdf_docs = pdf_loader.load()
        txt_docs = txt_loader.load()
        documents = pdf_docs + txt_docs
        
        if not documents:
            st.error(f"No documents (PDF or TXT) found in the '{DATA_PATH}' folder. Please add some files.")
            st.stop()
        else:
            st.write(f"Loaded {len(documents)} document(s).")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text could be extracted from the documents. Please ensure they are not empty or image-only.")
                st.stop()
            else:
                st.success(f"Split documents into {len(chunks)} chunks.")
                
                # Create and persist the vector database
                with st.spinner(f"Creating embeddings with '{EMBEDDING_MODEL}' and storing in Chroma..."):
                    try:
                        # FIX: Use the correct embedding model
                        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
                        db = Chroma.from_documents(
                            documents=chunks, 
                            embedding=embeddings, 
                            persist_directory=CHROMA_PATH
                        )
                        st.success("✅ Database created successfully!")
                    except Exception as e:
                        st.error(f"Error creating database: {e}")
                        st.error("Please ensure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull {EMBEDDING_MODEL}`)")
                        st.stop()
else:
    st.success(f"✅ Existing Chroma database found at '{CHROMA_PATH}'.")

# --- Query Section ---
query_text = st.text_input("Ask a question:", placeholder="What is Machine Learning?")

if query_text:
    with st.spinner("Searching the knowledge base and thinking..."):
        try:
            # Load the existing database
            # FIX: Must use the SAME embedding model to load the DB
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            retriever = db.as_retriever()
            
            # Initialize the LLM for answering
            # FIX: Use OllamaLLM for the main model
            model = OllamaLLM(model=LLM_MODEL) 

            # Create the prompt template
            prompt_template = """
            Answer the question based only on the following context:
            {context}
            ---
            Answer the question based on the above context: {question}
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create the RAG chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            # Invoke the chain and get the response
            start_time = time.time()
            response = chain.invoke(query_text)
            end_time = time.time()
            
            st.success(f"Here's the answer (generated in {end_time - start_time:.2f} seconds):")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(f"Make sure Ollama is running (`ollama serve`) and you have both models: `{LLM_MODEL}` and `{EMBEDDING_MODEL}`.")