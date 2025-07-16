import os
os.makedirs("batch_results", exist_ok=True)
import json
import logging
import fitz  # PyMuPDF
import docx
import torch
import psutil
import multiprocessing
import time
from functools import partial
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_results/generate_summaries.log"),
    ],
)
logger = logging.getLogger(__name__)


def read_pdf(file_path, extract_pages=True):
    try:
        text, pages = "", []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text().strip()
                text += page_text + "\n"
                if extract_pages and page_text:
                    pages.append({"number": page_num, "content": page_text})
        return text.strip(), pages
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return "", []


def read_docx(file_path, extract_pages=True, page_size=5):
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        pages = []
        if extract_pages:
            for i in range(0, len(paragraphs), page_size):
                pages.append({
                    "number": (i // page_size) + 1,
                    "content": "\n".join(paragraphs[i:i + page_size])
                })
        return full_text, pages
    except Exception as e:
        logger.error(f"Error reading DOCX {file_path}: {e}")
        return "", []


def load_summarization_model(model_dir):
    try:
        tokenizer = BartTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = BartForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device), tokenizer, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def generate_summary(text, model, tokenizer, device, max_length, min_length, num_beams):
    if len(text) < 50:
        return text
    try:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=num_beams,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return text[:200] + "..."


def check_system_performance():
    cpu_count = multiprocessing.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if gpu_available else 0

    return {
        'num_workers': max(1, cpu_count - 1),
        'batch_size': 200 if ram_gb >= 64 else 100 if ram_gb >= 32 else 50 if ram_gb >= 16 else 20,
        'max_length': 200 if gpu_memory_gb >= 8 else 150 if gpu_memory_gb >= 4 else 100,
        'min_length': 50 if gpu_memory_gb >= 8 else 30,
        'num_beams': 5 if gpu_memory_gb >= 8 else 4 if gpu_memory_gb >= 4 else 3,
        'docx_page_size': 10 if ram_gb >= 32 else 5
    }


def load_summarized_files(path):
    return set(json.load(open(path, "r", encoding="utf-8"))) if os.path.exists(path) else set()


def save_summarized_files(files_set, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(files_set), f, ensure_ascii=False, indent=2)


def process_batch(batch, folder_path, model_dir, batch_id, max_length, min_length, num_beams, docx_page_size):
    output_path = f"batch_results/books_batch_{batch_id}.json"
    model, tokenizer, device = load_summarization_model(model_dir)
    results = []
    for file in tqdm(batch, desc=f"Batch {batch_id}"):
        try:
            full_path = os.path.join(folder_path, file)
            if file.endswith(".pdf"):
                text, pages = read_pdf(full_path)
            elif file.endswith(".docx"):
                text, pages = read_docx(full_path, page_size=docx_page_size)
            else:
                continue
            summary = generate_summary(text, model, tokenizer, device, max_length, min_length, num_beams)
            results.append({"title": file, "summary": summary, "pages": pages})
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def process_documents(folder_path, model_dir, output_json="books.json", **kwargs):
    start = time.time()

    all_files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".docx"))]
    summarized = load_summarized_files("batch_results/summarized_files.json")
    new_files = [f for f in all_files if f not in summarized]

    if not new_files:
        logger.info("No new files to process.")
        return

    if kwargs.get("auto_params", True):
        sys_params = check_system_performance()
        for key in sys_params:
            kwargs[key] = kwargs.get(key) or sys_params[key]

    batches = [new_files[i:i + kwargs["batch_size"]] for i in range(0, len(new_files), kwargs["batch_size"])]
    args = [(batch, folder_path, model_dir, i + 1,
             kwargs["max_length"], kwargs["min_length"],
             kwargs["num_beams"], kwargs["docx_page_size"]) for i, batch in enumerate(batches)]

    with multiprocessing.Pool(kwargs["num_workers"]) as pool:
        batch_outputs = pool.starmap(process_batch, args)

    all_summaries = []
    for path in batch_outputs:
        with open(path, "r", encoding="utf-8") as f:
            all_summaries.extend(json.load(f))

    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            all_summaries = json.load(f) + all_summaries

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    summarized.update(new_files)
    save_summarized_files(summarized, "batch_results/summarized_files.json")
    logger.info(f"Processed {len(new_files)} files in {time.time() - start:.2f}s. Saved to {output_json}.")

# Entry point
if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        process_documents(
            folder_path="Books",
            model_dir="./bart-large-cnn",
            output_json="books.json",
            auto_params=True,
            max_length=None,
            min_length=None,
            batch_size=None,
            num_workers=None,
            num_beams=None,
            docx_page_size=None
        )
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)
