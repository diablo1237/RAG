"""
==============================================================================
  SIMPLE PDF Q&A SYSTEM  —  a tiny RAG app  (Day 7 hands-on project)
==============================================================================

This app lets you upload a PDF and ask questions about it. It is a complete
(but minimal) RAG system. The code below follows the EXACT same steps we drew
on the Day 7 architecture slide:

   PHASE 1 — INDEXING (done once, right after upload)
      1. LOAD    the PDF into text
      2. CHUNK   the text into small pieces
      3. EMBED   each chunk into a vector
      4. STORE   the vectors in a FAISS vector database

   PHASE 2 — QUERYING (done every time you ask a question)
      5. RETRIEVE the chunks most similar to your question
      6. AUGMENT  the prompt by pasting those chunks in as "context"
      7. GENERATE the final answer with the Gemini LLM

Tools used:
   - Streamlit            -> the chat user interface (the web page)
   - LangChain            -> glues the pipeline together
   - PyPDFLoader          -> reads the PDF (step 1)
   - Gemini Embeddings    -> makes the embeddings (steps 3 & 5)  <-- via the API
   - FAISS                -> stores & searches the vectors (steps 4 & 5)
   - Google Gemini        -> writes the answer (step 7)

This version uses Gemini for BOTH embedding and answering, so everything goes
through ONE API key and there is nothing to download. (The other version uses
a local Hugging Face model for embeddings instead.)

Run it with:   streamlit run app_gemini_embeddings.py
==============================================================================
"""

import os
import streamlit as st

# --- LangChain building blocks -----------------------------------------------
from langchain_community.document_loaders import PyPDFLoader          # step 1: LOAD
from langchain_text_splitters import RecursiveCharacterTextSplitter   # step 2: CHUNK
from langchain_google_genai import GoogleGenerativeAIEmbeddings     # step 3/5: EMBED (via Gemini API)
from langchain_community.vectorstores import FAISS                    # step 4/5: STORE + SEARCH
from langchain_google_genai import ChatGoogleGenerativeAI            # step 7: GENERATE


# A tiny helper: ask Google which models THIS key can use for generating text.
# This future-proofs the app — if Google retires a model name, students can just
# read the sidebar to see valid names instead of hitting a confusing 404 error.
def list_available_models():
    """Return the model names that support text generation for this API key."""
    try:
        import google.generativeai as genai
        names = []
        for m in genai.list_models():
            # keep only models that can actually answer prompts
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                names.append(m.name.replace("models/", ""))
        return names
    except Exception:
        return []  # if the listing fails, we just fall back to the default name


# =============================================================================
#  SETTINGS  —  change these in one place
# =============================================================================
EMBEDDING_MODEL = "gemini-embedding-001"  # Gemini's text embedding model (via API)
#   Like the chat model, embedding model names can change over time. If you get
#   a 404 for this, try "text-embedding-004" or check Google's embeddings docs.
LLM_MODEL       = "gemini-flash-latest" # stable alias -> always the current Flash model
#   Why "gemini-flash-latest"? Google keeps retiring specific version names
#   (gemini-1.5-flash, gemini-2.5-flash...) and closing them to new API keys.
#   This alias always points to the current Flash model, so it doesn't go stale.
#   The sidebar dropdown also lets you pick any model YOUR key supports.
CHUNK_SIZE      = 500                  # characters per chunk (Day 7, slide 11)
CHUNK_OVERLAP   = 100                  # characters shared between chunks
TOP_K           = 3                    # how many chunks to retrieve per question


# =============================================================================
#  PHASE 1  —  BUILD THE KNOWLEDGE BASE FROM THE PDF  (indexing)
# =============================================================================
# The @st.cache_resource line tells Streamlit: "only do this ONCE per PDF."
# Without it, the app would re-read and re-embed the whole PDF on every click.
@st.cache_resource(show_spinner=False)
def build_vector_database(pdf_path):
    """Turn a PDF file into a searchable FAISS vector database."""

    # --- STEP 1: LOAD -------------------------------------------------------
    # PyPDFLoader opens the PDF and returns its text, page by page.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # --- STEP 2: CHUNK ------------------------------------------------------
    # A 200-page PDF is too big to use at once, so we cut it into small pieces.
    # The overlap (100 chars) keeps context from being lost at the seams.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # --- STEP 3: EMBED ------------------------------------------------------
    # Gemini's embedding model converts each chunk of text into a vector
    # (a list of numbers that captures its meaning). This is an API call, so
    # it uses your Gemini key — no local model download needed.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # --- STEP 4: STORE ------------------------------------------------------
    # FAISS.from_documents does two things in one line: it embeds every chunk
    # and stores all the vectors in a FAISS database built for fast search.
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db, len(chunks)


# =============================================================================
#  PHASE 2  —  ANSWER A QUESTION USING THE DATABASE  (querying)
# =============================================================================
def answer_question(vector_db, llm, question):
    """Retrieve relevant chunks, build a prompt, and generate an answer."""

    # --- STEP 5: RETRIEVE ---------------------------------------------------
    # Turn the question into a vector and find the TOP_K most similar chunks.
    # (Under the hood: cosine similarity, exactly like Day 6.)
    relevant_chunks = vector_db.similarity_search(question, k=TOP_K)

    # Join the retrieved chunk texts together into one "context" block.
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    # --- STEP 6: AUGMENT ----------------------------------------------------
    # This is the heart of RAG: we PASTE the retrieved context into the prompt,
    # right above the question, and tell the model to answer using only it.
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # --- STEP 7: GENERATE ---------------------------------------------------
    # Send the augmented prompt to Gemini and get back the final answer.
    response = llm.invoke(prompt)

    # Pull out just the plain text. Newer Gemini models sometimes return the
    # answer as a LIST of blocks (each a dict with a "text" field) instead of a
    # simple string. This turns either form into clean text for the user.
    answer_text = response.content
    if isinstance(answer_text, list):
        answer_text = "".join(
            block.get("text", "") for block in answer_text
            if isinstance(block, dict)
        )

    # Return both the answer and the chunks (so we can show the sources).
    return answer_text, relevant_chunks


# =============================================================================
#  THE STREAMLIT USER INTERFACE  (the web page)
# =============================================================================
def main():
    st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
    st.title("📄 Chat with your PDF")
    st.caption("A simple RAG app — Load → Chunk → Embed → Store → Retrieve → Augment → Generate")

    # Chat history lives here. Set it up first so it always exists.
    if "messages" not in st.session_state:
        st.session_state.messages = []   # each item: {"role", "content", "sources"}

    # --- API KEY --------------------------------------------------------------
    # The app needs a free Google Gemini API key. The user pastes it in the
    # sidebar. (Get one at: https://aistudio.google.com/app/apikey)
    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Google API Key", type="password",
                                help="Get a free key at aistudio.google.com/app/apikey")
        st.markdown("1. Paste your API key\n2. Upload a PDF\n3. Ask questions!")
        # Let students wipe the conversation and start fresh.
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []

    if not api_key:
        st.info("👈 Paste your Google API key in the sidebar to begin.")
        return
    os.environ["GOOGLE_API_KEY"] = api_key

    # Once we have a key, show which models it can actually use, and let the
    # user pick. This avoids "model not found" 404s when Google updates names.
    with st.sidebar:
        available = list_available_models()
        if available:
            # Pick a sensible default that the key can actually use:
            #   1) our preferred alias if the key lists it,
            #   2) otherwise any "flash" model (cheap + free-tier friendly),
            #   3) otherwise just the first available model.
            if LLM_MODEL in available:
                default_index = available.index(LLM_MODEL)
            else:
                flash = [m for m in available if "flash" in m.lower()]
                default_index = available.index(flash[0]) if flash else 0
            chosen_model = st.selectbox("Gemini model", available, index=default_index)
        else:
            # Listing failed (rare) — fall back to the alias and hope for the best.
            chosen_model = LLM_MODEL
            st.caption("Couldn't list models automatically; using " + LLM_MODEL)

    # --- FILE UPLOAD ----------------------------------------------------------
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is None:
        st.info("👆 Upload a PDF to get started.")
        return

    # Save the uploaded file to disk so PyPDFLoader can read it.
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- BUILD THE DATABASE (Phase 1) ----------------------------------------
    with st.spinner("Reading and indexing your PDF... (this runs once)"):
        vector_db, num_chunks = build_vector_database(pdf_path)
    st.success(f"✅ Ready! Your PDF was split into {num_chunks} chunks and indexed.")

    # Create the Gemini LLM object once, using the model chosen in the sidebar.
    llm = ChatGoogleGenerativeAI(model=chosen_model, temperature=0)

    # --- CHAT (Phase 2, chatbot style) ---------------------------------------
    # Streamlit re-runs the whole script on every action, so we use
    # st.session_state (set up at the top of main) to "remember" the
    # conversation between questions.

    # 1) Re-draw the whole conversation so far (every past Q and A).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # If this was an answer, also show the chunks it used.
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"🔍 See the {len(msg['sources'])} chunks used"):
                    for i, chunk in enumerate(msg["sources"], start=1):
                        page = chunk.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} (page {page}):**")
                        st.write(chunk.page_content)
                        st.divider()

    # 2) The chat input box pinned at the bottom of the page.
    question = st.chat_input("Ask a question about your PDF...")
    if question:
        # Show the user's question immediately and save it to history.
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get the answer (retrieve -> augment -> generate) and show it.
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and thinking..."):
                answer, sources = answer_question(vector_db, llm, question)
            st.write(answer)
            with st.expander(f"🔍 See the {len(sources)} chunks used"):
                for i, chunk in enumerate(sources, start=1):
                    page = chunk.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} (page {page}):**")
                    st.write(chunk.page_content)
                    st.divider()

        # Save the answer (and its sources) to history.
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()