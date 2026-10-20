
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# 1. Find all PDFs
data_folder = Path("data/raw")
pdf_files = sorted(data_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF found in data/raw")

print("PDF files found:", len(pdf_files))

# 2. Connect to the existing ChromaDB database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="university_documents"
)

# Chunk settings
chunk_size = 500
overlap = 100
batch_size = 20

processed_since_pause = 0
first_request = True

# 3. Process every PDF
for pdf_path in pdf_files:
    print(f"\nProcessing PDF: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))
    text = ""

    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    if not text.strip():
        print("No extractable text. Skipping this PDF.")
        continue

    # Split into chunks
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    print("Chunks created:", len(chunks))

    # 4. Embed and store missing chunks
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]

        ids = [
            f"{pdf_path.name}_chunk_{batch_start + i}"
            for i in range(len(batch))
        ]

        existing_ids = set(collection.get(ids=ids)["ids"])

        pending = [
            (chunk_id, chunk)
            for chunk_id, chunk in zip(ids, batch)
            if chunk_id not in existing_ids
        ]

        if not pending:
            print("Batch already stored. Skipping.")
            continue

        if first_request:
            print("Waiting 45 seconds for quota reset...")
            time.sleep(45)
            first_request = False

        if processed_since_pause >= 80:
            print("Pausing 65 seconds for quota...")
            time.sleep(65)
            processed_since_pause = 0

        pending_ids = [item[0] for item in pending]
        pending_chunks = [item[1] for item in pending]

        print(
            f"Embedding chunks {batch_start + 1}"
            f"–{batch_start + len(batch)} of {len(chunks)}"
        )

        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=pending_chunks,
                config={"task_type": "RETRIEVAL_DOCUMENT"}
            )

            embeddings = [
                item.values for item in response.embeddings
            ]

            collection.upsert(
                ids=pending_ids,
                documents=pending_chunks,
                embeddings=embeddings,
                metadatas=[
                    {"source": pdf_path.name}
                    for _ in pending_chunks
                ]
            )

            processed_since_pause += len(pending_chunks)
            print("Stored successfully.")

        except Exception as e:
            print("\nEMBEDDING ERROR:", type(e).__name__)
            print("Details:", str(e))
            raise

print("\nAll PDFs processed!")
print("Total chunks in database:", collection.count())
