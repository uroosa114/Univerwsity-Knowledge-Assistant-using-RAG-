
import os

from dotenv import load_dotenv
from google import genai
import chromadb

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# Connect to the existing vector database
db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_collection(
    name="university_documents"
)

# Ask the user for a question
question = input(
    "Ask a question about your MCA syllabus: "
).strip()

if not question:
    raise ValueError("Please enter a question.")

# Convert the question into an embedding
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config={"task_type": "RETRIEVAL_QUERY"}
)

query_embedding = response.embeddings[0].values

# Retrieve the 4 most relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4,
    include=["documents", "metadatas"]
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

# Combine retrieved chunks into context
context = "\n\n".join(
    f"Source: {metadata.get('source', 'Unknown')}\n{document}"
    for document, metadata in zip(documents, metadatas)
)

# Ask Gemini to answer using the retrieved context
prompt = f"""
You are a University Knowledge Assistant.

Answer the question using ONLY the context provided below.
Do not invent information.

If the answer is not available in the context, say:
"I couldn't find that information in the provided syllabus."

Context:
{context}

Question:
{question}

Give a clear, student-friendly answer.
Mention the source document.
"""

answer = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print("\n--- ANSWER ---")
print(answer.text)

print("\n--- RETRIEVED SOURCE DOCUMENTS ---")
for metadata in metadatas:
    print("-", metadata.get("source", "Unknown"))
