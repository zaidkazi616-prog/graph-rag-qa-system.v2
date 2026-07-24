import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from PyPDF2 import PdfReader

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Graph RAG QA", layout="centered")
st.title("Multi-Document QA System")
st.caption("Powered by Graph RAG (Retrieval Augmented Generation) with PDF support")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    return embedder, qa_tokenizer, qa_model

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text

def process_documents(doc_dict):
    chunks = []
    chunk_id = 0
    for doc_name, text in doc_dict.items():
        sentences = text.replace("\n", " ").strip().split(". ")
        for sentence in sentences:
            if len(sentence) > 15: # Filter out tiny fragments
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
            # Connect chunks if they share at least 3 common words
            # Increased from 2 to 3 to make graph connections more meaningful
            if len(common_words) >= 3:
                graph[i].append(j)
                graph[j].append(i)
    return graph

# --- SESSION STATE ---
# We use session state to store user-uploaded data so it persists across interactions
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
                # Load models
                embedder, qa_tokenizer, qa_model = load_models()
                
                # Extract text from PDF
                raw_text = extract_text_from_pdf(uploaded_file)
                
                # Process into chunks and build graph
                doc_dict = {uploaded_file.name: raw_text}
                new_chunks = process_documents(doc_dict)
                
                if len(new_chunks) == 0:
                    st.error("Could not extract enough text from this PDF. Please try another file.")
                else:
                    texts = [c["text"] for c in new_chunks]
                    new_embeddings = embedder.encode(texts)
                    new_graph = build_graph(new_chunks)
                    
                    # Save to session state
                    st.session_state.chunks = new_chunks
                    st.session_state.embeddings = new_embeddings
                    st.session_state.graph = new_graph
                    st.session_state.embedder = embedder
                    st.session_state.qa_tokenizer = qa_tokenizer
                    st.session_state.qa_model = qa_model
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
            
            # Step C: Generate Answer
            inputs = st.session_state.qa_tokenizer(question, final_context, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = st.session_state.qa_model(**inputs)
                
            answer_start = outputs.start_logits.argmax()
            answer_end = outputs.end_logits.argmax() + 1
            answer = st.session_state.qa_tokenizer.decode(inputs.input_ids[0, answer_start:answer_end])
            
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