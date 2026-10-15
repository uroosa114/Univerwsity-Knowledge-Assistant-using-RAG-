
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Explain RAG in simple words in 3 sentences."
        )

        print(response.text)

    except Exception as e:
        print("\nAPI ERROR TYPE:", type(e).__name__)
        print("API ERROR DETAILS:", str(e))
