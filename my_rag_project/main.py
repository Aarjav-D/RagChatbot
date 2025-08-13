# from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import glob
import requests
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from any origin (adjust for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load docs and build vector store from PDFs and TXTs
docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

for file in glob.glob("content/*"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file)
    elif file.endswith(".txt"):
        loader = TextLoader(file)
    else:
        print(f"Skipping unsupported file: {file}")
        continue

    file_docs = loader.load()
    docs.extend(splitter.split_documents(file_docs))

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# HuggingFace API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/openai/gpt-oss-20b"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Make sure this env var is set!

system_prompt = """
You are a strict QA bot.
Answer ONLY from the context provided.
If the answer is not in the context, reply exactly: "I don't know."
"""

class QueryRequest(BaseModel):
    question: str

def call_llm_chat(system_prompt, question, context):
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0,
            "max_new_tokens": 512,
            "return_full_text": False
        }
    }

    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"HuggingFace API error: {response.text}")

    data = response.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    else:
        raise HTTPException(status_code=500, detail="Unexpected response from HuggingFace API")

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    question = req.question
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    answer = call_llm_chat(system_prompt, question, context_text)
    return {"answer": answer}
# from pydantic import BaseModel
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# import glob
# import requests
# import os
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # Add CORS middleware to allow frontend requests from cxneo.com or localhost during testing
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to your frontend domain in prod e.g. ["https://cxneo.com"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load docs and build vector store (or load persisted db)
# docs = []
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# for file in glob.glob("content/*"):
#     loader = TextLoader(file)
#     file_docs = loader.load()
#     docs.extend(splitter.split_documents(file_docs))

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# db = FAISS.from_documents(docs, embeddings)
# retriever = db.as_retriever(search_kwargs={"k": 3})

# # HuggingFace Inference API config
# HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-27b-it?inference_provider=nebius"
# HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Set this env var before running!

# system_prompt = """
# You are a strict QA bot.
# Answer ONLY from the context provided.
# If the answer is not in the context, reply exactly: "I don't know."
# """

# class QueryRequest(BaseModel):
#     question: str

# def call_llm_chat(system_prompt, question, context):
#     prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

#     headers = {
#         "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
#     }
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "temperature": 0,
#             "max_new_tokens": 512,
#             "return_full_text": False
#         }
#     }

#     response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
#     if response.status_code != 200:
#         raise HTTPException(status_code=500, detail=f"HuggingFace API error: {response.text}")

#     data = response.json()
#     # Extract generated text safely
#     if isinstance(data, list) and "generated_text" in data[0]:
#         return data[0]["generated_text"].strip()
#     else:
#         raise HTTPException(status_code=500, detail="Unexpected response from HuggingFace API")

# @app.post("/chat")
# async def chat_endpoint(req: QueryRequest):
#     question = req.question
#     docs = retriever.get_relevant_documents(question)
#     context_text = "\n\n".join([doc.page_content for doc in docs])
#     answer = call_llm_chat(system_prompt, question, context_text)
#     return {"answer": answer}
# from fastapi import FastAPI
# from pydantic import BaseModel
# import requests
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from fastapi import FastAPI, HTTPException
# import glob

# app = FastAPI()

# # Load docs and build vector store (or you can load persisted db)
# docs = []
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# for file in glob.glob("content/*"):
#     loader = TextLoader(file)
#     file_docs = loader.load()
#     docs.extend(splitter.split_documents(file_docs))

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# db = FAISS.from_documents(docs, embeddings)
# retriever = db.as_retriever(search_kwargs={"k": 3})

# LM_API = "http://localhost:1234/v1"
# MODEL_ID = "mistral-7b-instruct-v0.3"

# system_prompt = """
# You are a strict QA bot.
# Answer ONLY from the context provided.
# If the answer is not in the context, reply exactly: "I don't know."
# """

# class QueryRequest(BaseModel):
#     question: str

# def call_llm_chat(system_prompt, question, context):
#     messages = [
#         {"role": "user", "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
#     ]

#     payload = {
#         "model": MODEL_ID,
#         "messages": messages,
#         "temperature": 0,
#         "max_tokens": 512
#     }
    
#     print("Sending payload to LM Studio:")
#     print(payload)
    
#     response = requests.post(f"{LM_API}/chat/completions", json=payload)
    
#     print("LM Studio response status:", response.status_code)
#     print("LM Studio response text:", response.text)
    
#     response.raise_for_status()
    
#     return response.json()["choices"][0]["message"]["content"]

# @app.post("/chat")
# async def chat_endpoint(req: QueryRequest):
#     try:
#         question = req.question
#         docs = retriever.get_relevant_documents(question)
#         context_text = "\n\n".join([doc.page_content for doc in docs])
#         answer = call_llm_chat(system_prompt, question, context_text)
#         return {"answer": answer}
#     except Exception as e:
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))