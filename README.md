# 🎓 University Knowledge Assistant using RAG

An AI-powered university knowledge assistant that helps students find academic information by asking questions in natural language. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from university documents and generate context-aware answers.

🚀 **Live Demo:** [Try the University Knowledge Assistant](https://share.streamlit.io/)

## 📌 Project Overview

University students often need to search through multiple documents to find information about academic calendars, course topics, examinations, and other university-related queries.

The University Knowledge Assistant simplifies this process by allowing students to ask questions and receive answers based on the information available in university documents.

Instead of relying entirely on a language model's existing knowledge, the application retrieves relevant document content and uses it to generate more relevant responses.

## ✨ Features

- **AI-Powered Question Answering:** Ask university-related questions in natural language.
- **RAG Architecture:** Retrieves relevant information from university documents before generating answers.
- **Document-Based Responses:** Uses indexed university documents as the knowledge source.
- **Vector Database:** Stores document embeddings using ChromaDB for similarity-based retrieval.
- **Google Gemini Integration:** Uses Gemini to generate responses from retrieved context.
- **Interactive Web Interface:** Built with Streamlit for a simple, user-friendly experience.
- **Cloud Deployment:** Deployed using Streamlit Community Cloud.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web application interface |
| LangChain | Document processing and RAG components |
| ChromaDB | Vector storage and similarity search |
| Google Gemini API | AI-generated answers |
| PDF Processing | Extracting text from university documents |
| GitHub | Source code management and deployment integration |

## ⚙️ How It Works

The project follows a Retrieval-Augmented Generation workflow.

1. **Document Ingestion:** University PDF documents are loaded and their text is extracted.
2. **Text Chunking:** The extracted text is split into smaller chunks for efficient retrieval.
3. **Embedding Generation:** Text chunks are converted into vector embeddings.
4. **Vector Storage:** The embeddings and associated document content are stored in ChromaDB.
5. **Question Processing:** The user submits a question through the Streamlit interface.
6. **Context Retrieval:** Relevant document chunks are retrieved using similarity search.
7. **Answer Generation:** The retrieved context is passed to Google Gemini to generate an answer.
8. **Response Display:** The generated answer is displayed to the user through the web interface.

## 🏗️ Project Structure

```text
University-Knowledge-Assistant-using-RAG/
│
├── app.py                 # Streamlit application
├── ingestion.py           # PDF ingestion and vector database creation
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git
├── .env                   # Local environment variables (not committed)
├── university_documents/   # University PDF documents, if included
└── README.md              # Project documentation
```

*Note: Adjust the folder structure above to match the actual files in your repository.*

## 💻 Installation and Setup

### Prerequisites

- Python 3.9 or a compatible version supported by your dependencies
- Git
- A Google Gemini API key

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd University-Knowledge-Assistant-using-RAG
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Never commit API keys or other secrets to GitHub.

### 5. Process University Documents

Run the document ingestion script:

```bash
python3 ingestion.py
```

This step extracts document content, creates embeddings, and stores the indexed data in ChromaDB.

### 6. Run the Application

```bash
python3 -m streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## ☁️ Deployment

The application is deployed on **Streamlit Community Cloud** and connected to a GitHub repository.

To deploy your own instance:

1. Push your project files to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository.
4. Choose `app.py` as the main application file.
5. Add `GEMINI_API_KEY` under the application's Secrets settings.
6. Ensure your university documents and ChromaDB data are accessible in the deployed environment.
7. Deploy and test the application.

**Important:** If your ChromaDB database is created locally, make sure the deployment process also creates or loads
