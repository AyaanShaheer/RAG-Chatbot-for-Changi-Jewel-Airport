# main.py

import os
from dotenv import load_dotenv

import gradio as gr # <-- Import Gradio
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- INITIALIZATION ---
load_dotenv()

if os.environ.get("GOOGLE_API_KEY") is None:
    print("❌ Google API key not found. Please set it in the .env file.")
    exit()

# FastAPI App remains the main app
app = FastAPI(
    title="Changi Airport Chatbot API",
    description="A RAG-based chatbot for Changi Airport and Jewel Changi websites.",
    version="1.0.0",
)


# --- LANGCHAIN RAG SETUP (No changes here) ---
print("🧠 Loading knowledge base...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.1
)

prompt_template = """
Answer the following question based on the context provided. If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("✅ Knowledge base loaded and RAG chain ready.")


# --- GRADIO UI SETUP (New Section) ---

def chatbot_response(message, history):
    """
    This is the core function that Gradio will call.
    It takes a user message and chat history, gets a response
    from our RAG chain, and returns the answer.
    """
    print(f"Gradio Query: {message}")
    result = qa_chain.invoke({"query": message})
    return result["result"]

# Create the Gradio Chat UI
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Changi Airport & Jewel Chatbot",
    description="Ask me questions about Changi Airport and Jewel Changi. I'll answer based on their official websites.",
    theme="soft",
    examples=[
        "What dining options are available at Jewel?",
        "Where is the butterfly garden located?",
        "How can I claim my GST refund?"
    ]
)

# Mount the Gradio app onto the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/ui")


# --- API ENDPOINTS (No changes here) ---

class QueryRequest(BaseModel):
    question: str

@app.get("/", summary="Root endpoint for API status check.")
def read_root():
    return {"status": "ok", "message": "API is running. Access the Chat UI at /ui"}

@app.post("/ask", summary="Ask a question via API.")
def ask_question(query: QueryRequest):
    print(f"API Query: {query.question}")
    result = qa_chain.invoke({"query": query.question})
    return {
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }