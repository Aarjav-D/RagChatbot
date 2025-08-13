from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import glob

# 1. Load all files from content/
docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

for file in glob.glob("content/*"):
    loader = TextLoader(file)
    file_docs = loader.load()
    docs.extend(splitter.split_documents(file_docs))

# 2. Create FAISS vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# 3. Create retriever
retriever = db.as_retriever(search_kwargs={"k": 7})

# 4. Custom prompt
prompt_template = """
You are a strict QA bot.
Answer ONLY from the context below.
If the answer is not in the context, reply exactly: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 5. Local llama.cpp model
llm = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0
)

while True:
    q = input("\nAsk: ")
    if q.lower() == "exit":
        break

    # Get top chunks from FAISS
    retrieved_docs = retriever.get_relevant_documents(q)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Run LLM with only your context
    final_prompt = prompt.format(context=context_text, question=q)
    answer = llm.invoke(final_prompt)

    print("\n💡 Answer:", answer)
    print("\n📄 Retrieved from:")
    for doc in retrieved_docs:
        print(f" - {doc.metadata.get('source', 'unknown')}")