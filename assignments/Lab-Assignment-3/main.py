# 1. Imports
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# 2. Load Environment Variables
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# 3. Load Document
loader = PyPDFLoader("data/document.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# 4. Split Into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# 5. Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 6. Create Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 7. Initialize LLM (Groq)
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

# 8. Build RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 9. CLI Interaction
query = input("\nAsk a question about the document: ")

response = qa_chain.invoke({"query": query})

print("\nAnswer:\n")
print(response["result"])

print("\nRetrieved Context:\n")
for doc in response["source_documents"]:
    print(doc.page_content[:300])
    print("-" * 50)
