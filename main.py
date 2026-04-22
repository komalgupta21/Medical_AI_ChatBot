# ==========================================
# 🏥 MediBot FINAL (Stable Version)
# ==========================================

# 🔹 Install required packages (force reinstall)
!pip install -q --upgrade \
langchain \
langchain-community \
langchain-text-splitters \
langchain-huggingface \
langchain-google-genai \
faiss-cpu \
pypdf \
sentence-transformers \
transformers

# ==========================================
# 🔑 API KEY (PUT YOUR KEY)
# ==========================================
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_REAL_API_KEY"

# ==========================================
# 📦 IMPORTS
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI

# Try PDF imports safely
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# ==========================================
# 📄 LOAD PDF (OPTIONAL SAFE MODE)
# ==========================================
retriever = None
pdf_path = "Full.pdf"

if PDF_AVAILABLE and os.path.exists(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        print("✅ PDF loaded successfully")
    except Exception as e:
        print("⚠️ PDF disabled:", e)
        retriever = None
else:
    print("⚠️ Running without PDF (safe mode)")

# ==========================================
# 🤖 GEMINI MODEL (100% WORKING)
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",   # ✅ MOST STABLE MODEL
    temperature=0.2
)

# ==========================================
# 🚨 EMERGENCY CHECK
# ==========================================
def is_emergency(text):
    keywords = ["chest pain", "stroke", "can't breathe", "bleeding heavily"]
    return any(k in text.lower() for k in keywords)

# ==========================================
# 🧠 MEDIBOT FUNCTION
# ==========================================
def medibot(query):

    if is_emergency(query):
        return "🚨 Emergency! Please go to hospital immediately."

    context = ""

    # Use PDF if available
    if retriever and len(query.split()) > 3:
        try:
            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])
        except:
            pass

    prompt = f"""
You are a medical assistant.

Question: {query}

Context: {context}

Answer in:
- Overview
- Symptoms
- Causes
- Treatment
- When to see doctor
"""

    try:
        response = llm.invoke(prompt)
        return response.content + "\n\n⚕️ Disclaimer: Consult a doctor."
    except Exception as e:
        return f"⚠️ Model Error: {e}"

# ==========================================
# 💬 CHAT LOOP
# ==========================================
print("\n💬 MediBot Ready! Type 'exit'\n")

while True:
    q = input("🧑 You: ")

    if q.lower() == "exit":
        print("👋 Goodbye!")
        break

    print("\n🤖 MediBot:\n")
    print(medibot(q))
    print("\n" + "-"*50)
