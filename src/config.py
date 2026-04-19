import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nemotron-3-nano:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# "auto" tries Ollama first, falls back to OpenRouter, then DeepSeek.
# Set to "ollama", "openrouter", or "deepseek" to force a specific backend.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
OPEN_ROUTER_MODEL = os.getenv("OPEN_ROUTER_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
TEST_PATH = os.getenv("TEST_PATH", "data/test.csv")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
