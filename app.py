import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Graph RAG QA", layout="centered")
st.title("Multi-Document QA System")
st.caption("Powered by Graph RAG (Retrieval Augmented Generation) with PDF support")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Using a generative model that writes real answers
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, tokenizer, model

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}. The file might be corrupted or scanned as an image.")
        return None

def process_documents(doc_dict):
    chunks = []
    chunk_id = 0
    for doc_name, text in doc_dict.items():
        sentences = text.replace("\n", " ").strip().split(". ")
        for sentence in sentences:
            if len(sentence) > 15:
                chunks.append({"id": chunk_id, "text": sentence, "source": doc_name})
                chunk_id += 1
    return chunks

def build_graph(chunks):
    graph = {i: [] for i in range(len(chunks))}
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            words_i = set(chunks[i]["text"].lower().split())
            words_j = set(chunks[j]["text"].lower().split())
            common_words = words_i.intersection(words_j)
            if len(common_words) >= 3:
                graph[i].append(j)
                graph[j].append(i)
    return graph

# --- SESSION STATE ---
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'graph' not in st.session_state:
    st.session_state.graph = {}
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    if st.button("Build Graph RAG System"):
        if uploaded_file is not None:
            with st.spinner("Extracting text and building knowledge graph..."):
                embedder, tokenizer, model = load_models()
                raw_text = extract_text_from_pdf(uploaded_file)
                
                if raw_text is None or len(raw_text.strip()) == 0:
                    st.error("Could not extract text from this PDF. It might be an image-based PDF without text layers.")
                else:
                    doc_dict = {uploaded_file.name: raw_text}
                    new_chunks = process_documents(doc_dict)
                    
                    if len(new_chunks) == 0:
                        st.error("Could not extract enough text from this PDF. Please try another file.")
                    else:
                        texts = [c["text"] for c in new_chunks]
                        new_embeddings = embedder.encode(texts)
                        new_graph = build_graph(new_chunks)
                        
                        st.session_state.chunks = new_chunks
                        st.session_state.embeddings = new_embeddings
                        st.session_state.graph = new_graph
                        st.session_state.embedder = embedder
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model = model
                        st.session_state.system_ready = True
                        
                        st.success(f"Successfully processed {uploaded_file.name}! You can now ask questions.")
        else:
            st.warning("Please upload a PDF file first.")

# --- MAIN APP LOGIC ---
if not st.session_state.system_ready:
    st.info("Please upload a PDF document in the sidebar and click 'Build Graph RAG System' to begin.")
else:
    st.success("System is ready. Ask a question based on your uploaded document.")
    
    question = st.text_input("Enter your question:", placeholder="e.g., What is the main topic of the document?")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching Graph & Generating Answer..."):
            # Step A: Vector Search
            q_embedding = st.session_state.embedder.encode([question])[0]
            similarities = np.dot(st.session_state.embeddings, q_embedding)
            best_chunk_id = np.argmax(similarities)
            
            # Step B: Graph Traversal
            context_chunks = [st.session_state.chunks[best_chunk_id]["text"]]
            sources_used = [st.session_state.chunks[best_chunk_id]["source"]]
            
            for neighbor_id in st.session_state.graph[best_chunk_id]:
                context_chunks.append(st.session_state.chunks[neighbor_id]["text"])
                sources_used.append(st.session_state.chunks[neighbor_id]["source"])
            
            final_context = " ".join(context_chunks)
            
            # Step C: Generate Answer using Flan-T5
            prompt = f"Context: {final_context}\nQuestion: {question}\nAnswer:"
            inputs = st.session_state.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = st.session_state.model.generate(**inputs, max_new_tokens=50)
                
            answer = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display Results
            st.markdown("### Answer:")
            if answer.strip():
                st.info(answer)
            else:
                st.warning("The model could not find a direct answer in the retrieved context. Try rephrasing your question.")
            
            st.markdown("### Sources Retrieved from Graph:")
            for src in set(sources_used):
                st.markdown(f"- `{src}`")

# Footer
st.markdown("---")
st.markdown("Built for College Project | Graph RAG Architecture Version 2")
