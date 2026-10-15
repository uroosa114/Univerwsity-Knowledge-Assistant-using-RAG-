
import os
import csv

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

# File where experiment results will be saved
output_file = "comparison_results.csv"

print("RAG vs Conventional LLM Experiment")
print("Enter a syllabus question.")
print("Press Enter without typing a question to finish.\n")

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Question",
        "Conventional LLM Answer",
        "RAG Answer",
        "Retrieved Sources"
    ])

    while True:
        question = input("\nYour question: ").strip()

        if not question:
            break

        print("\nRunning conventional LLM...")

        # SYSTEM A: No syllabus retrieval
        baseline_prompt = f"""
        Answer the following question using your general knowledge.
        You do not have access to the university syllabus.
        If you are uncertain, say so rather than inventing facts.

        Question: {question}
        """

        baseline_response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=baseline_prompt
        )

        baseline_answer = baseline_response.text or ""

        print("\nRunning RAG system...")

        # Convert the question into an embedding
        embedding_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question,
            config={"task_type": "RETRIEVAL_QUERY"}
        )

        query_embedding = embedding_response.embeddings[0].values

        # Retrieve relevant syllabus chunks
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            include=["documents", "metadatas"]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        context = "\n\n".join(documents)

        sources = sorted({
            metadata.get("source", "Unknown")
            for metadata in metadatas
        })

        # SYSTEM B: Answer using retrieved context
        rag_prompt = f"""
        You are a University Knowledge Assistant.

        Answer using ONLY the context below.
        Do not invent information.

        If the answer is not available in the context, say:
        "I couldn't find that information in the provided syllabus."

        Context:
        {context}

        Question: {question}

        Give a clear, student-friendly answer.
        """

        rag_response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=rag_prompt
        )

        rag_answer = rag_response.text or ""

        # Display both answers
        print("\n--- CONVENTIONAL LLM ANSWER ---")
        print(baseline_answer)

        print("\n--- RAG ANSWER ---")
        print(rag_answer)

        print("\n--- RETRIEVED SOURCE(S) ---")
        for source in sources:
            print("-", source)

        # Save the experiment results
        writer.writerow([
            question,
            baseline_answer,
            rag_answer,
            "; ".join(sources)
        ])

        file.flush()

        print("\nResults saved to", output_file)

print("\nExperiment session completed.")
print("Results are in comparison_results.csv")
