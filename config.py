"""
Central configuration for the AI Student Assistant.

Every script imports its paths and settings from here, so you only
change things in one place. Values can be overridden with a `.env` file
(copy `.env.example` to `.env`).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = ROOT_DIR / "chroma_db"

RAW_FILE = RAW_DIR / "educationq_full.json"
CLEAN_CSV = PROCESSED_DIR / "educationq_clean.csv"
CLEAN_JSONL = PROCESSED_DIR / "educationq_clean.jsonl"

# -------------------------------------------------------------- dataset
HF_DATASET = "SunriserFuture/EducationQ"
HF_FILE = "educationq_full.json"  # contains MMLU-Pro (1300) + GPQA (198)
DATASET_URL = "https://huggingface.co/datasets/SunriserFuture/EducationQ"

# ----------------------------------------------------------- embeddings
# Multilingual Sentence Transformer for English, French, and Arabic (384 dimensions).
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Keep the multilingual index separate from the old English-only index.
COLLECTION_NAME = "educationq_multilingual"

# ------------------------------------------------------------ retrieval
TOP_K = int(os.getenv("TOP_K", 4))
# Cosine distance in [0, 2]; 0 = identical. Results farther than this are
# treated as "not relevant". Tune it with step 6 (evaluation).
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", 0.65))

# ------------------------------------------------------------------ LLM
# Any OpenAI-compatible API works: Groq (free tier), OpenAI, or a local
# Ollama server. See README "Step 5".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_PROVIDER_DEFAULTS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
}
_default_url, _default_model = _PROVIDER_DEFAULTS.get(LLM_PROVIDER, (None, None))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", _default_url)
LLM_MODEL = os.getenv("LLM_MODEL", _default_model)
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama" if LLM_PROVIDER == "ollama" else "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
