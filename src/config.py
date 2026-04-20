import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nemotron-3-nano:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# "auto" tries Ollama first, then OpenRouter.
# Set to "ollama", "openrouter", or "huggingface" to force a backend.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "nvidia/OpenMath-Nemotron-14B-Kaggle")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
TEST_PATH = os.getenv("TEST_PATH", "data/test.csv")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
