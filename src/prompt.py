cat > /c/Users/kg643/Medical_AI_ChatBot/src/prompt.py << 'EOF'
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the given context to answer the question. If you don't know the answer, say you don't know. Context: {context}"),
    ("human", "{input}")
])
EOF