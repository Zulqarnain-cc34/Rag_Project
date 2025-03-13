# RAG: Retrieval-Augmented Generation System

This project implements an Retrieval-Augmented Generation (RAG) system that processes various file types and leverages advanced retrieval strategies for query answering. The system extracts text from documents (Markdown and PowerPoint files, and applies retrieval strategies to produce AI-generated responses using state-of-the-art language models.

---

## Features

- **File Processing**:
  - Supports Markdown (`.md`) and PowerPoint (`.pptx`) files.
  - Uses custom extractors for different file formats to create unified document representations.

- **Built with Modern Tools**:
  - **LangChain**: For prompt handling, chaining, and LLM integrations.
  - **Pg_vector**: For vector-based similarity search.
  - **OpenAI Models**: For text generation and embedding.
  - **Pydantic**: For structured data validation.

---

## Workflow

1. **File Processing & Extraction**:
   - Files in the designated directory (`./data/unprocessed_docs`) are processed.
   - Markdown and PPTX files are parsed using dedicated extractor classes.
   
2. **Document Creation & Embedding**:
   - Extracted text is split into manageable chunks.
   - Each chunk is converted to embeddings using OpenAI embeddings.
   - A pg_vector index is built for efficient similarity searches.

3. **Retrieval**:
   - The system invokes the retrieval strategy specified.
   
4. **Response Generation**:
   - Retrieved documents are combined and passed as context to a language model.
   - The model generates a final answer using a prompt template.

---

## Requirements

- **Python 3.10+** (tested on Python 3.12)
- Environment variables set for OpenAI API (see `.env` file)
- Required Python packages:
  - `langchain`, `langchain_openai`, `langchain_core`
  - `langchain-postgres`
  - `pydantic`
  - `python-dotenv`
  - `pptx` (for processing PowerPoint files)

---

## Setting up PG_Vector 

You can run the following command to spin up a a postgres container with the pgvector extension:

        docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16


## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Umbrage-Studios/umbrage-rag.git
   cd umbrage-rag
   ```

2. **Create and Configure Environment Variables**

   - Copy the example environment file and update it with your OpenAI API key:
   
     ```bash
     cp .env.example .env
     ```
     
   - Edit the `.env` file and set the `OPENAI_API_KEY` value.

3. **Install Dependencies**

   It is recommended to use a virtual environment:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Running the System

1. **Prepare Your Documents**

   - Place the documents you wish to process in the `./data/unprocessed_docs` directory.
   - Supported formats include Markdown (`.md`) and PowerPoint (`.pptx`) files.

2. **Run the Adaptive RAG System**

   The main entry point is `adaptive_retreival.py`. Run it with:

   ```bash
   python adaptive_retreival.py
   ```

   This script will:
   - Process the files in the provided directory.
   - Build a vector store from extracted text.
   - Classify and answer a preset query (modify or extend as needed).

3. **Custom Query Processing**

   You can modify the list of queries in `adaptive_retreival.py` or extend the system to accept command-line inputs for dynamic querying.

---

## Folder Structure

```
.
├── apps
│   ├── backend
│   │   ├── src
│   │   │   ├── data                 # Contains input documents
│   │   │   ├── processor            # Houses file processors and extractor files
│   │   │   │   ├── base_extractor.py
│   │   │   │   ├── markdown_extractor.py
│   │   │   │   ├── pptx_extractor.py
│   │   │   │   └── file_processor.py
│   │   │   ├── strategy             # Contains adaptive retrieval strategy implementation
│   │   │   │   └── adaptive_strategy.py
│   │   │   ├── main.py              # Entry point to run file processing and Adaptive RAG system
│   │   │   └── .env.example         # Example environment configuration file
│   │   │   └── .env                 # Environment file
│   │   │   └── README.md            # Backend specific documentation
│   │   └── requirements.txt         # Backend Python dependencies
│   └── frontend                     # Frontend code (if applicable)
└── 
```

- **apps/backend/src/data/**: Contains unprocessed input documents.
- **apps/backend/src/processor/**: Houses file processors and extractor files.
- **apps/backend/src/strategy/**: Contains the adaptive retrieval strategy implementation.
- **apps/backend/src/main.py**: Orchestrates the file processing and Adaptive RAG query answering.
- **apps/frontend/**: Contains the frontend code (if applicable).

## Development Workflow

1. **Create a New Branch**

   - Branch off from `main` (or your designated development branch) using a descriptive name:
   
     ```bash
     git checkout -b feature/add-new-extractor
     ```

2. **Implement & Test**

   - Make code changes, add unit tests, and verify functionality locally.
   - Follow coding standards and commit messages following Conventional Commits.

3. **Push & Create a Pull Request**

   - Push your branch and open a pull request for review.
   - Include a clear description of your changes and reference any related issues.

---

## Additional Notes

- **Logging & Debugging**: Logging is integrated into the extractor modules for easier troubleshooting.
- **Extensibility**: New file extractors can be added by extending the `BaseExtractor` class and updating the `FileProcessor` extractor mapping.
- **Error Handling**: The system handles file read errors gracefully and logs issues for unsupported file types.

---

## Troubleshooting

- **API Key Issues**: Ensure your `.env` file is correctly configured with a valid OpenAI API key.
- **Dependency Errors**: Verify that all required packages are installed. Use a virtual environment to avoid conflicts.
- **File Extraction**: Check that the files in the `./data/unprocessed_docs` folder are in a supported format (.md, .pptx). Logs will indicate any extraction issues.

---

Happy coding and creating umbrage Slack RAG Bot!
