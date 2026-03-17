import os
from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings

load_dotenv()

app = Flask(__name__)

# ── Setup ─────────────────────────────────────────────────
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    api_key=os.environ.get("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the given context to answer the question. If you don't know the answer, say you don't know. Context: {context}"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke(msg)
    return str(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)