"""Debug: print raw LLM planner output for one puzzle per type."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["LLM_PROVIDER"] = "deepseek"

import pandas as pd
from src.llm_client import LLMClient
from src.planner import PLANNER_SYSTEM
from src.classify import PUZZLE_SIGNATURES
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

llm = LLMClient(provider="deepseek", deepseek_api_key=DEEPSEEK_API_KEY, deepseek_model=DEEPSEEK_MODEL)
df = pd.read_csv("data/train.csv")

for sig, ptype in PUZZLE_SIGNATURES.items():
    row = df[df["prompt"].str.contains(sig, case=False, na=False)].iloc[0]
    print(f"\n{'='*70}")
    print(f"TYPE: {ptype}  ID: {row['id']}")
    print(f"{'='*70}")

    resp = llm.chat(
        [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"PUZZLE_TYPE: {ptype}\n\nPROMPT:\n{row['prompt']}"},
        ],
        think=False, temperature=0.3, max_tokens=8192,
    )
    raw = (resp.content or "").strip()
    print(raw)
    print(f"\n--- length: {len(raw)} chars ---")
