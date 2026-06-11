import os
import uuid
import shutil
import tempfile
import gc
from pathlib import Path
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")
st.write("Upload one or more PDFs and ask questions using Retrieval-Augmented Generation.")


groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.markdown(
        """
        <div style="padding: 1rem; border: 1px solid #ffb3b3; border-radius: 0.5rem; background-color: #fff5f5;">
            <strong>GROQ_API_KEY not found.</strong><br>
            Locally, add it in your .env file. On Hugging Face Spaces, add it in Settings → Variables and secrets.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_session_base_path():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    base_dir = Path(tempfile.gettempdir()) / "pdf_rag_chroma"
    session_path = base_dir / st.session_state.session_id
    session_path.mkdir(parents=True, exist_ok=True)

    return session_path


def safe_delete_folder(folder_path):
    try:
        folder_path = Path(folder_path)

        if folder_path.exists():
            shutil.rmtree(folder_path)

    except PermissionError:
        pass

    except OSError:
        pass


def clear_vectorstore():
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore

    gc.collect()

    session_path = get_session_base_path()
    safe_delete_folder(session_path)

    keys_to_delete = [
        "pdf_processed",
        "file_names",
        "total_pdfs",
        "total_pages",
        "total_chunks",
        "chroma_path",
        "processed_chunk_size",
        "processed_chunk_overlap"
    ]

    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]


def load_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        pdf_documents = loader.load()

        os.remove(temp_path)

        for doc in pdf_documents:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(pdf_documents)

    return documents


def split_documents(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = get_embeddings()

    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
        gc.collect()

    session_path = get_session_base_path()
    index_id = str(uuid.uuid4())
    chroma_path = session_path / index_id
    chroma_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
        collection_name="pdf_rag_collection"
    )

    st.session_state.chroma_path = str(chroma_path)

    return vectorstore


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=groq_api_key
    )


def get_style_instruction(answer_style):
    if answer_style == "Simple explanation":
        return "Explain the answer in simple and easy student-friendly language."

    if answer_style == "Exam notes":
        return "Write the answer like short exam notes with important exam points."

    if answer_style == "Detailed answer":
        return "Give a detailed answer with clear explanation."

    if answer_style == "Bullet points":
        return "Answer using clean bullet points."

    return "Answer clearly and simply."


def get_query_mode_instruction(query_mode):
    if query_mode == "Compare PDFs / Find mentions":
        return """
The user is asking in multi-PDF comparison mode.

Focus on:
1. Which uploaded PDF files mention the topic.
2. What each relevant PDF says about the topic.
3. Source file names and page numbers.
4. Differences between PDFs when possible.

Important:
If a PDF is not found in the retrieved context, say that it was not found in the retrieved context.
Do not claim that the PDF definitely does not contain the topic unless the retrieved context proves it.
"""

    return """
The user is asking in normal Q&A mode.
Answer the question directly using the retrieved PDF context.
"""


def get_recent_conversation(messages, max_turns):
    if not messages or max_turns <= 0:
        return ""

    recent_messages = messages[-max_turns * 2:]
    formatted_history = []

    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {message['content']}")

    return "\n".join(formatted_history)


def normalize_relevance_score(score):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


def get_relevance_label(score):
    score = normalize_relevance_score(score)

    if score >= 0.75:
        return "High"

    if score >= 0.50:
        return "Medium"

    return "Low"


def retrieve_documents(question, vectorstore, top_k, conversation_context):
    if conversation_context.strip():
        retrieval_query = (
            f"Recent conversation:\n{conversation_context}\n\n"
            f"Current question:\n{question}"
        )
    else:
        retrieval_query = question

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            retrieval_query,
            k=top_k
        )

        return [
            {
                "doc": doc,
                "score": normalize_relevance_score(score),
                "label": get_relevance_label(score)
            }
            for doc, score in results
        ]

    except Exception:
        results = vectorstore.similarity_search_with_score(
            retrieval_query,
            k=top_k
        )

        converted_results = []

        for doc, distance in results:
            relevance_score = 1 / (1 + float(distance))
            converted_results.append(
                {
                    "doc": doc,
                    "score": normalize_relevance_score(relevance_score),
                    "label": get_relevance_label(relevance_score)
                }
            )

        return converted_results


def build_context(retrieved_results):
    context_parts = []

    for item in retrieved_results:
        doc = item["doc"]
        score = item["score"]
        label = item["label"]

        context_parts.append(
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 0) + 1}, "
            f"Relevance: {label}, "
            f"Relevance Score: {score:.2f}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def build_source_overview(retrieved_results):
    source_map = defaultdict(lambda: {"pages": set(), "max_score": 0.0})

    for item in retrieved_results:
        doc = item["doc"]
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        score = item["score"]

        source_map[source]["pages"].add(page)
        source_map[source]["max_score"] = max(source_map[source]["max_score"], score)

    if not source_map:
        return "No relevant sources were retrieved."

    overview_lines = []

    for source, data in source_map.items():
        pages = sorted(data["pages"])
        pages_text = ", ".join(str(page) for page in pages)
        max_score = data["max_score"]
        label = get_relevance_label(max_score)

        overview_lines.append(
            f"{source}: pages {pages_text}, highest relevance {label} ({max_score:.2f})"
        )

    return "\n".join(overview_lines)


def answer_question(
    question,
    vectorstore,
    answer_style,
    query_mode,
    top_k,
    memory_turns,
    previous_messages
):
    conversation_context = get_recent_conversation(
        previous_messages,
        memory_turns
    )

    retrieved_results = retrieve_documents(
        question=question,
        vectorstore=vectorstore,
        top_k=top_k,
        conversation_context=conversation_context
    )

    context = build_context(retrieved_results)
    source_overview = build_source_overview(retrieved_results)

    style_instruction = get_style_instruction(answer_style)
    query_mode_instruction = get_query_mode_instruction(query_mode)

    uploaded_files = "\n".join(
        st.session_state.get("file_names", [])
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a helpful PDF question-answering assistant.

Rules:
1. Answer only using the provided PDF context.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
   "I could not find this information in the uploaded PDF."
4. Mention the source file name and page number when possible.
5. Use recent conversation only to understand follow-up questions.
6. Do not invent sources or page numbers.
7. {style_instruction}

Query mode instructions:
{query_mode_instruction}
"""
            ),
            (
                "human",
                """
Uploaded PDF files:
{uploaded_files}

Recent conversation:
{conversation_context}

Question:
{question}

Retrieved source overview:
{source_overview}

PDF Context:
{context}
"""
            )
        ]
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "uploaded_files": uploaded_files,
            "conversation_context": conversation_context,
            "question": question,
            "source_overview": source_overview,
            "context": context
        }
    )

    return response.content, retrieved_results


def render_message(role, content):
    if role == "user":
        st.markdown("**You:**")
    else:
        st.markdown("**Assistant:**")

    st.write(content)


def format_chat_history_for_download():
    lines = ["PDF RAG Chatbot Chat History", ""]

    for message in st.session_state.get("messages", []):
        role = "You" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}:")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines)


def render_empty_state():
    st.markdown(
        """
        <div style="
            border: 1px solid #d0d7de;
            border-radius: 12px;
            padding: 24px;
            background-color: #f8fafc;
            margin-top: 16px;
        ">
            <h3 style="margin-top: 0;">No PDFs uploaded yet</h3>
            <p>Upload one or more PDFs from the sidebar, choose your chunk settings, and click <strong>Process PDFs</strong>.</p>
            <p>After processing, you can ask questions, compare PDFs, view source pages, and download your chat history.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def handle_question(question):
    previous_messages = st.session_state.messages.copy()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    render_message("user", question)

    with st.spinner("Searching PDFs and generating answer..."):
        answer, retrieved_results = answer_question(
            question=question,
            vectorstore=st.session_state.vectorstore,
            answer_style=st.session_state.answer_style,
            query_mode=st.session_state.query_mode,
            top_k=st.session_state.top_k,
            memory_turns=st.session_state.memory_turns,
            previous_messages=previous_messages
        )

    render_message("assistant", answer)

    with st.expander("Sources used"):
        for i, item in enumerate(retrieved_results, start=1):
            doc = item["doc"]
            score = item["score"]
            label = item["label"]

            source_name = doc.metadata.get("source", "Unknown")
            page_number = doc.metadata.get("page", 0) + 1

            st.markdown(f"### Source {i}")
            st.write(f"**File:** {source_name}")
            st.write(f"**Page:** {page_number}")
            st.write(f"**Relevance:** {label}")
            st.write(f"**Relevance score:** {score:.2f}")
            st.write(doc.page_content[:1000])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "Simple explanation"

if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Normal Q&A"

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "memory_turns" not in st.session_state:
    st.session_state.memory_turns = 3


with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.header("RAG Settings")

    st.session_state.chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=2000,
        value=st.session_state.chunk_size,
        step=100,
        help="Larger chunks keep more context. Smaller chunks can retrieve more precise text."
    )

    st.session_state.chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=500,
        value=st.session_state.chunk_overlap,
        step=50,
        help="Overlap keeps continuity between chunks but increases duplicate text."
    )

    if st.session_state.chunk_overlap >= st.session_state.chunk_size:
        st.write("Chunk overlap must be smaller than chunk size.")

    st.session_state.top_k = st.slider(
        "Top K retrieved chunks",
        min_value=2,
        max_value=8,
        value=st.session_state.top_k,
        step=1,
        help="Higher values retrieve more context but may include less relevant chunks."
    )

    st.session_state.memory_turns = st.slider(
        "Conversation memory turns",
        min_value=0,
        max_value=5,
        value=st.session_state.memory_turns,
        step=1,
        help="Number of recent conversation turns used to understand follow-up questions."
    )

    st.header("Answer Settings")

    st.session_state.answer_style = st.selectbox(
        "Choose answer style",
        [
            "Simple explanation",
            "Exam notes",
            "Detailed answer",
            "Bullet points"
        ]
    )

    st.session_state.query_mode = st.selectbox(
        "Choose query mode",
        [
            "Normal Q&A",
            "Compare PDFs / Find mentions"
        ]
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF(s) uploaded")

        if st.button("Process PDFs"):
            if st.session_state.chunk_overlap >= st.session_state.chunk_size:
                st.write("Reduce chunk overlap before processing PDFs.")
            else:
                with st.spinner("Reading PDFs, creating chunks, and building vector database..."):
                    documents = load_pdfs(uploaded_files)
                    chunks = split_documents(
                        documents=documents,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    vectorstore = create_vectorstore(chunks)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.total_pdfs = len(uploaded_files)
                    st.session_state.total_pages = len(documents)
                    st.session_state.total_chunks = len(chunks)
                    st.session_state.file_names = [file.name for file in uploaded_files]
                    st.session_state.processed_chunk_size = st.session_state.chunk_size
                    st.session_state.processed_chunk_overlap = st.session_state.chunk_overlap
                    st.session_state.messages = []

                st.write("PDFs processed successfully.")
                st.write(f"Total PDFs: {st.session_state.total_pdfs}")
                st.write(f"Total pages: {st.session_state.total_pages}")
                st.write(f"Total chunks: {st.session_state.total_chunks}")

    st.divider()

    st.header("Sample Questions")

    sample_questions = [
        "Summarize the uploaded PDF.",
        "What are the main topics covered?",
        "Explain this in simple words.",
        "Give important exam points.",
        "Create short notes from this PDF.",
        "Which PDF mentions the main topic?"
    ]

    for sample in sample_questions:
        if st.button(sample, disabled="vectorstore" not in st.session_state):
            st.session_state.sample_question = sample

    st.divider()

    if st.session_state.messages:
        chat_text = format_chat_history_for_download()

        st.download_button(
            label="Download chat history",
            data=chat_text,
            file_name="pdf_rag_chat_history.txt",
            mime="text/plain"
        )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear PDFs and Vector DB"):
        st.session_state.messages = []
        clear_vectorstore()
        st.rerun()


if "vectorstore" not in st.session_state:
    render_empty_state()
else:
    st.write("PDFs are ready. Ask your questions below.")

    with st.expander("Processed PDFs"):
        for file_name in st.session_state.get("file_names", []):
            st.write(f"📄 {file_name}")

        st.write(f"Total PDFs: {st.session_state.get('total_pdfs', 0)}")
        st.write(f"Total pages: {st.session_state.get('total_pages', 0)}")
        st.write(f"Total chunks: {st.session_state.get('total_chunks', 0)}")
        st.write(f"Chunk size used: {st.session_state.get('processed_chunk_size', 0)}")
        st.write(f"Chunk overlap used: {st.session_state.get('processed_chunk_overlap', 0)}")

    for message in st.session_state.messages:
        render_message(message["role"], message["content"])

    user_question = st.chat_input("Ask a question from your PDFs...")

    trigger_question = st.session_state.pop("sample_question", None) or user_question

    if trigger_question:
        handle_question(trigger_question)