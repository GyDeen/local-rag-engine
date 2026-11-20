from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_ROOT = PROJECT_ROOT / "examples/docs"
DEFAULT_ERROR_CATALOG = PROJECT_ROOT / "examples/catalog.json"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Generation defaults; environment variables can override deployment settings.
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT = 300
DEFAULT_GENERATION_CONCURRENCY = 1
DEFAULT_GENERATION_QUEUE_SIZE = 32

DEFAULT_DATABASE_BACKEND = "chroma"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data/vector_db"
# The collection name records the embedder, because vectors from different
# models are not interchangeable.
DEFAULT_COLLECTION_NAME = "docs_bge_m3_c1000_o100"


def validate_evidence_rank_limit(value: int | None) -> None:
    """An unset cutoff preserves the full stitching pool."""
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("evidence_rank_limit must be a positive integer or None")


# Evidence selection strategies accepted by retrieval.
RERANKERS = ("none", "cross_encoder", "llm")

# More files reduced retrieval quality in the evidence-budget experiment.
DEFAULT_FILE_LIMIT = 3

# Characters of stitched documentation handed to answer generation.
DEFAULT_EVIDENCE_CHAR_BUDGET = 6000

# Sections whose stitched text is shorter than this are not evidence.
DEFAULT_EVIDENCE_MIN_CHUNK_CHARS = 200
