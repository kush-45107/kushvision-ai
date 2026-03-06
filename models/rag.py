import os
import base64
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None
uploaded_image_path = None


# File Process
def process_file(file):
    global vectorstore, uploaded_image_path

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    ext = file.filename.split(".")[-1].lower()

    # Image Store
    if ext in ["png", "jpg", "jpeg", "webp"]:
        uploaded_image_path = path
        vectorstore = None  # reset document memory
        return "Image uploaded successfully. Now ask your question."

    # Document Process
    uploaded_image_path = None  # reset image memory

    if ext == "pdf":
        loader = PyPDFLoader(path)
    elif ext == "txt":
        loader = TextLoader(path, encoding="utf-8")
    elif ext == "docx":
        loader = Docx2txtLoader(path)
    elif ext == "csv":
        loader = CSVLoader(path)
    else:
        return "Unsupported file"

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return "File uploaded successfully. Now ask your question."


# Ask Function
def ask_file(question):
    global vectorstore, uploaded_image_path

    # Image
    if uploaded_image_path:

        with open(uploaded_image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

        chat = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                }
            ],
        )

        return chat.choices[0].message.content

    # Document
    if vectorstore:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])

        final_prompt = f"""
You are a helpful AI.
Answer ONLY from given context.
If answer not found say: Not found in file.

Context:
{context}

Question:
{question}
"""

        chat = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": final_prompt}]
        )

        return chat.choices[0].message.content

    return "Upload a file first."