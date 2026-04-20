"""Baseline accuracy across all 4 equation_transform subtypes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
from src.tools import solve_equation_transform
from src.classify import classify_equation_subtype

SUBTYPES = [
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "cryptarithm_deduce",
    "cryptarithm_guess",
]

def run():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]

    total_correct = total_wrong = total_skipped = 0
    for subtype in SUBTYPES:
        subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == subtype)]
        correct = wrong = skipped = 0
        for _, r in subset.iterrows():
            try:
                res = solve_equation_transform(json.dumps({"prompt": r["prompt"]}))
                if str(res).strip() == str(r["answer"]).strip():
                    correct += 1
                else:
                    wrong += 1
            except Exception:
                skipped += 1
        total = correct + wrong + skipped
        pct = correct / total * 100 if total else 0
        print(f"{subtype:<30} {correct:>4}/{total:<4} = {pct:5.1f}%  wrong={wrong} skipped={skipped}")
        total_correct += correct
        total_wrong += wrong
        total_skipped += skipped

    grand = total_correct + total_wrong + total_skipped
    grand_pct = total_correct / grand * 100 if grand else 0
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_correct:>4}/{grand:<4} = {grand_pct:5.1f}%  wrong={total_wrong} skipped={total_skipped}")

if __name__ == "__main__":
    run()
