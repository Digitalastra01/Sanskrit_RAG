# Sanskrit RAG System

A Retrieval-Augmented Generation (RAG) system designed for Sanskrit documents, capable of ingesting text, retrieving relevant context, and generating responses using a local CPU-optimized LLM.

## Features

- **Sanskrit Support**: Specialized handling of Sanskrit text for ingestion and retrieval.
- **Local Inference**: Runs entirely on CPU using `llama.cpp` and quantized models, ensuring data privacy and offline capability.
- **Efficient Retrieval**: Uses FAISS for vector storage and `sentence-transformers` for multilingual embeddings.
- **Interactive UI**: Streamlit-based interface for easy querying.
- **Modern Tooling**: Managed with `uv` for fast and reliable dependency management.

## Prerequisites

- Python 3.12+
- `uv` package manager

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RAG
    ```

2.  **Initialize and Sync Dependencies:**
    ```bash
    uv sync
    ```

3.  **Download Model:**
    Downloads `TinyLlama-1.1B-Chat-v1.0-GGUF` (quantized) to `models/`.
    ```bash
    uv run python code/download_model.py
    ```

## Usage

1.  **Ingest Data:**
    Process Sanskrit stories from `data/` and create the vector store.
    ```bash
    uv run python code/ingest.py
    ```

2.  **Run the Application:**
    Launch the Streamlit interface.
    ```bash
    uv run streamlit run code/app.py
    ```

3.  **Query:**
    Open the browser URL (usually `http://localhost:8501`) and enter your question in Sanskrit or English (e.g., "कुरुक्षेत्रे के समवेताः आसन्?").

## Project Structure

- `code/`: Source code for the application.
    - `app.py`: Streamlit frontend.
    - `ingest.py`: Data ingestion and vector store creation.
    - `rag_pipeline.py`: RAG logic (retrieval and generation).
    - `download_model.py`: Script to download the LLM.
- `data/`: Directory for input text files.
- `models/`: Directory for GGUF models.
- `vectorstore/`: FAISS index storage.
- `report/`: Technical documentation.
- `pyproject.toml`: Project configuration and dependencies.
