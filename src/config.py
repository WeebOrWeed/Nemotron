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

# Local HuggingFace/PEFT planner inference. This is how the locally trained
# LoRA adapter is exercised by main.py when PLANNER_PROVIDER=hf_lora.
HF_PLANNER_BASE_MODEL = os.getenv("HF_PLANNER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
HF_PLANNER_LORA_PATH = os.getenv("HF_PLANNER_LORA_PATH", "models/planner-lora")
HF_PLANNER_MAX_NEW_TOKENS = int(os.getenv("HF_PLANNER_MAX_NEW_TOKENS", "192"))
HF_PLANNER_LOAD_4BIT = os.getenv("HF_PLANNER_LOAD_4BIT", "1").lower() not in {
    "0", "false", "no", "off",
}

# Force the planner to use a specific backend. If unset, classify._make_planner_llm
# picks the best available cloud LLM (OpenRouter, then DeepSeek). Set to
# "ollama" to use the local model for planning too (slower DAG quality but free),
# or "hf_lora" to use the local PEFT adapter at HF_PLANNER_LORA_PATH.
PLANNER_PROVIDER = os.getenv("PLANNER_PROVIDER", "")

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
TEST_PATH = os.getenv("TEST_PATH", "data/test.csv")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
