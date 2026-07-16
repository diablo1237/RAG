import os
import streamlit as st

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "gemini-flash-latest"

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# List Gemini models
def list_available_models():
    try:
        import google.generativeai as genai
        names = []
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                names.append(m.name.replace("models/", ""))
        return names
    except Exception:
        return []


# Build vector DB
@st.cache_resource(show_spinner=False)
def build_vector_database(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db, len(chunks)


# Answer question
def answer_question(vector_db, llm, question):
    relevant_chunks = vector_db.similarity_search(question, k=TOP_K)

    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = f"""
You are a helpful assistant. Answer ONLY from the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    # Handle Gemini structured output
    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "") for block in answer if isinstance(block, dict)
        )

    return answer, relevant_chunks


# Main app
def main():
    st.set_page_config(page_title="PDF Q&A", page_icon="📄")
    st.title("📄 Chat with your PDF")

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("Setup")

        api_key = st.text_input("Google API Key", type="password")

        st.markdown("""
1. Paste API key  
2. Upload PDF  
3. Ask questions  
""")

        if st.button("Clear chat"):
            st.session_state.messages = []

    if not api_key:
        st.info("Enter your Google API key to continue")
        return

    os.environ["GOOGLE_API_KEY"] = api_key

    # Model selection
    available = list_available_models()

    if available:
        if LLM_MODEL in available:
            default_index = available.index(LLM_MODEL)
        else:
            flash_models = [m for m in available if "flash" in m.lower()]
            default_index = available.index(flash_models[0]) if flash_models else 0

        chosen_model = st.sidebar.selectbox(
            "Select Gemini model", available, index=default_index
        )
    else:
        chosen_model = LLM_MODEL
        st.sidebar.caption(f"Using default model: {LLM_MODEL}")

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is None:
        st.info("Upload a PDF to begin")
        return

    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Build vector DB
    with st.spinner("Processing PDF..."):
        vector_db, num_chunks = build_vector_database(pdf_path)

    st.success(f"PDF processed into {num_chunks} chunks")

    # LLM
    llm = ChatGoogleGenerativeAI(model=chosen_model, temperature=0)

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"See {len(msg['sources'])} source chunks"):
                    for i, chunk in enumerate(msg["sources"], start=1):
                        page = chunk.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page})**")
                        st.write(chunk.page_content)
                        st.divider()

    # Chat input
    question = st.chat_input("Ask a question about your PDF")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(vector_db, llm, question)

            st.write(answer)

            with st.expander(f"See {len(sources)} source chunks"):
                for i, chunk in enumerate(sources, start=1):
                    page = chunk.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} (page {page})**")
                    st.write(chunk.page_content)
                    st.divider()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()