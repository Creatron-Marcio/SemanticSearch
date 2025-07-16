# SemanticSearch

A FastAPI-based semantic search service for book summaries using FAISS vector database and Sentence Transformers.

## Features

- Semantic search across book summaries using embeddings
- Fast vector similarity search with FAISS
- RESTful API with FastAPI
- Document processing pipeline for PDFs and DOCX files
- Automatic text summarization with BART
- Page-level search to find the most relevant page for a query
- Automatic parameter optimization based on system capabilities
- Scalable architecture for processing millions of books
- Parallel processing with multiprocessing
- Pagination support for large result sets
- Memory-efficient processing with batching

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── Preprocessing/          # Data preparation scripts
│   ├── faiss_indexer.py    # Creates FAISS index from book summaries
│   ├── generate_summaries.py # Generates summaries from books
│   ├── all-MiniLM-L6-v2/   # Sentence transformer model
│   └── bart-large-cnn/     # Text summarization model
├── README.md               # Main documentation
├── ADD_NEW_BOOKS.md        # Guide for adding new books to existing index
├── AUTO_PARAMS.md          # Documentation for automatic parameter optimization
├── LARGE_SCALE.md          # Guide for processing large datasets
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv env`
3. Activate the environment: 
   - Windows: `env\Scripts\activate`
   - Linux/Mac: `source env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Documentation

- [ADD_NEW_BOOKS.md](ADD_NEW_BOOKS.md) - Instructions for adding new books to existing index
- [AUTO_PARAMS.md](AUTO_PARAMS.md) - Details on automatic parameter optimization
- [LARGE_SCALE.md](LARGE_SCALE.md) - Guide for processing large datasets

## Usage

### Preprocessing

1. Place your PDF/DOCX documents in the `Preprocessing/Books/` directory
2. Generate summaries with automatic parameter optimization:
   ```bash
   python Preprocessing/generate_summaries.py
   ```
   The script will automatically detect your system's capabilities and set optimal parameters.
   
   You can also manually specify parameters:
   ```bash
   python Preprocessing/generate_summaries.py --batch-size 100 --workers 4 --no-auto
   ```

3. Create FAISS index with automatic parameter optimization:
   ```bash
   python Preprocessing/faiss_indexer.py
   ```
   
   Or manually specify parameters:
   ```bash
   python Preprocessing/faiss_indexer.py --batch-size 1000 --use-ivf 10000 --nprobe 16 --no-auto
   ```

### Running the API

#### Development Mode
```bash
uvicorn main:app --reload
```

#### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using Docker
```bash
docker-compose up -d
```

The API will be available at http://localhost:8000 and the API documentation at http://localhost:8000/docs

### API Endpoints

- POST `/search` - Search for books by semantic similarity
  - Request body: `{"query": "your search query", "top_k": 5, "max_score": 1.0, "page": 1, "page_size": 10}`
  - Response includes the most relevant page from each document matching the query
- GET `/health` - Health check endpoint

## License

MIT