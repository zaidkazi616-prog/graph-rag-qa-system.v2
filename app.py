import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import pdfplumber

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Graph RAG QA", layout="centered")
st.title("Multi-Document QA System")
st.caption("Powered by Graph RAG (Retrieval Augmented Generation) with Llama 3.1")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

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
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            sentences = para.replace("\n", " ").strip().split(". ")
            current_chunk = ""
            for sentence in sentences:
                if len(sentence) > 15:
                    if len(current_chunk) == 0:
                        current_chunk = sentence
                    else:
                        current_chunk += ". " + sentence
                    
                    if current_chunk.count(". ") >= 2:
                        chunks.append({"id": chunk_id, "text": current_chunk, "source": doc_name})
                        chunk_id += 1
                        current_chunk = ""
            if len(current_chunk) > 15:
                chunks.append({"id": chunk_id, "text": current_chunk, "source": doc_name})
                chunk_id += 1
    return chunks

def build_graph(chunks):
    graph = {i: [] for i in range(len(chunks))}
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            words_i = set(chunks[i]["text"].lower().split())
            words_j = set(chunks[j]["text"].lower().split())
            common_words = words_i.intersection(words_j)
            if len(common_words) >= 4:
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
                embedder = load_models()
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
                        st.session_state.system_ready = True
                        
                        st.success(f"Successfully processed {uploaded_file.name}! You can now ask questions.")
        else:
            st.warning("Please upload a PDF file first.")

# --- MAIN APP LOGIC ---
if not st.session_state.system_ready:
    st.info("Please upload a PDF in the sidebar to begin.")
else:
    st.success("System is ready. Ask a question based on your uploaded document.")
    
    question = st.text_input("Enter your question:", placeholder="e.g., What is the main topic of the document?")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching Graph & Generating Answer with Llama 3.1..."):
            # Step A: Vector Search
            q_embedding = st.session_state.embedder.encode([question])[0]
            similarities = np.dot(st.session_state.embeddings, q_embedding)
            best_chunk_id = np.argmax(similarities)
            
            # Step B: Graph Traversal
            context_chunks = [st.session_state.chunks[best_chunk_id]["text"]]
            sources_used = [st.session_state.chunks[best_chunk_id]["source"]]
            
            neighbor_count = 0
            for neighbor_id in st.session_state.graph[best_chunk_id]:
                if neighbor_count < 3:
                    context_chunks.append(st.session_state.chunks[neighbor_id]["text"])
                    sources_used.append(st.session_state.chunks[neighbor_id]["source"])
                    neighbor_count += 1
            
            final_context = " ".join(context_chunks)
            
            # Step C: Generate Answer using Groq (Llama 3.1)
            try:
                # This reads the key from the hidden Streamlit vault
                client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                
                prompt = f"Context: {final_context}\n\nQuestion: {question}\n\nAnswer the question based strictly on the context provided. If the answer is not in the context, say 'I cannot find the answer in the document.'"
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions strictly based on the provided context."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    # UPDATED MODEL NAME HERE
                    model="llama-3.1-8b-instant",
                )
                
                answer = chat_completion.choices[0].message.content
                
                # Display Results
                st.markdown("### Answer:")
                st.info(answer)
                
                st.markdown("### Sources Retrieved from Graph:")
                for src in set(sources_used):
                    st.markdown(f"- `{src}`")
                    
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# Footer
st.markdown("---")
st.markdown("Built for College Project | Graph RAG Architecture Version 2")
