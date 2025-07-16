import json
import os
import logging
import faiss
import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("faiss_indexer.log"),
    ],
)
logger = logging.getLogger(__name__)

def check_system_performance():
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if gpu_available else 0

    logger.info(f"System Info — RAM: {ram_gb:.1f} GB, GPU: {gpu_available}, GPU RAM: {gpu_memory_gb:.1f} GB")
    
    return {
        'batch_size': 5000 if ram_gb >= 64 else 2000 if ram_gb >= 32 else 1000 if ram_gb >= 16 else 500,
        'use_ivf': 10000 if ram_gb >= 32 else 20000,
        'nprobe': 16 if ram_gb >= 32 else 10
    }

def load_books(json_path):
    if not os.path.exists(json_path):
        logger.error(f"Input file does not exist: {json_path}")
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        books = json.load(f)
    logger.info(f"Loaded {len(books)} books from {json_path}")
    return books

def build_faiss_index(
    books, model_path, index_path, meta_path,
    batch_size=None, use_ivf=None, nprobe=None, auto_params=True
):
    if auto_params:
        sys_params = check_system_performance()
        batch_size = batch_size or sys_params['batch_size']
        use_ivf = use_ivf or sys_params['use_ivf']
        nprobe = nprobe or sys_params['nprobe']
    else:
        batch_size = batch_size or 1000
        use_ivf = use_ivf or 10000
        nprobe = nprobe or 10

    model = SentenceTransformer(model_path)
    logger.info(f"Model loaded: {model_path}")

    # Sample embedding
    sample_text = f"{books[0]['title']} {books[0]['summary']}"
    dim = model.encode([sample_text], convert_to_numpy=True).shape[1]
    logger.info(f"Embedding dimension: {dim}")

    use_ivf_index = len(books) > use_ivf
    if use_ivf_index:
        n_centroids = max(min(4 * int(len(books)**0.5), len(books)//10), 100)
        logger.info(f"Using IndexIVFFlat with {n_centroids} centroids")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_centroids)
        index.nprobe = nprobe
    else:
        logger.info("Using IndexFlatL2")
        index = faiss.IndexFlatL2(dim)

    if use_ivf_index:
        train_size = min(100000, len(books))
        train_texts = [f"{books[i]['title']} {books[i]['summary']}" for i in np.random.choice(len(books), train_size, replace=False)]
        train_embeddings = model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
        index.train(train_embeddings)
        logger.info("Index trained")

    for i in range(0, len(books), batch_size):
        batch = books[i:i+batch_size]
        texts = [f"{b['title']} {b['summary']}" for b in batch]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        index.add(embeddings)
        logger.info(f"Processed batch {i // batch_size + 1}")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info(f"Index saved to {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False)
    logger.info(f"Metadata saved to {meta_path}")

import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Build FAISS index from book summaries")
    parser.add_argument("--input", "-i", default="books.json", help="Input JSON file with book summaries")
    parser.add_argument("--model", "-m", default="./all-MiniLM-L6-v2", help="Path to SentenceTransformer model")
    parser.add_argument("--output", "-o", default="books.index", help="Output path for FAISS index")
    parser.add_argument("--meta", default="books_meta.json", help="Output path for metadata JSON")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing large datasets")
    parser.add_argument("--use-ivf", type=int, help="Threshold for using IVF index (dataset size)")
    parser.add_argument("--nprobe", type=int, help="Number of centroids to visit during search")
    parser.add_argument("--no-auto", action="store_true", help="Disable automatic parameter detection")
    return parser.parse_args()

# --- Run with CLI args ---
if __name__ == "__main__":
    try:
        args = parse_args()
        books = load_books(args.input)
        build_faiss_index(
            books,
            model_path=args.model,
            index_path=args.output,
            meta_path=args.meta,
            batch_size=args.batch_size,
            use_ivf=args.use_ivf,
            nprobe=args.nprobe,
            auto_params=not args.no_auto
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        exit(1)
