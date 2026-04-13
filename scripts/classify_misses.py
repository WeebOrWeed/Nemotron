"""Classify prediction misses by puzzle type and write per-type markdown reports."""
import pandas as pd
import os

pred = pd.read_csv(r"C:\Users\zhaot\Downloads\predictions.csv")
train = pd.read_csv("data/train.csv")


def answers_match(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-2 + 1e-9
    except (ValueError, TypeError):
        return False


pred["match"] = pred.apply(lambda r: answers_match(r["answer"], r["expected"]), axis=1)
misses = pred[~pred["match"]].copy()
merged = misses.merge(train[["id", "prompt"]], on="id", how="left")

SIGS = {
    "bit manipulation": "bit_manipulation",
    "numeral system": "numeral_conversion",
    "unit conversion": "unit_conversion",
    "encryption rules": "cipher_decryption",
    "transformation rules": "equation_transform",
    "gravitational constant": "gravity_physics",
}


def classify(prompt):
    p = str(prompt).lower()
    for sig, ptype in SIGS.items():
        if sig in p:
            return ptype
    return "unknown"


merged["puzzle_type"] = merged["prompt"].apply(classify)

# Per-type counts in the full train set
train_merged = pred.merge(train[["id", "prompt"]], on="id", how="left")
train_merged["puzzle_type"] = train_merged["prompt"].apply(classify)
type_totals = train_merged["puzzle_type"].value_counts().to_dict()
type_matches = train_merged[train_merged["match"]].groupby("puzzle_type").size().to_dict()

os.makedirs("results/misses", exist_ok=True)

for ptype, group in merged.groupby("puzzle_type"):
    total = type_totals.get(ptype, "?")
    matched = type_matches.get(ptype, 0)
    miss_count = len(group)
    acc = matched / total * 100 if isinstance(total, int) and total > 0 else 0

    lines = []
    lines.append(f"# {ptype} misses ({miss_count} total)\n\n")
    lines.append(f"**Type stats:** {matched} match / {total} total = **{acc:.1f}% accuracy**\n\n")
    lines.append("---\n\n")

    for _, r in group.iterrows():
        lines.append(f"## id: {r['id']}\n\n")
        lines.append(f"**Expected:** `{r['expected']}`  \n")
        lines.append(f"**Actual:** `{r['answer']}`  \n\n")
        prompt_text = str(r["prompt"])
        if len(prompt_text) > 600:
            prompt_text = prompt_text[:600] + "..."
        lines.append(f"<details><summary>Prompt (click to expand)</summary>\n\n```\n{prompt_text}\n```\n\n</details>\n\n---\n\n")

    path = f"results/misses/{ptype}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  {ptype}: {miss_count} misses, {acc:.1f}% accuracy -> {path}")

TYPE_ORDER = ["unit_conversion", "gravity_physics", "bit_manipulation", "equation_transform"]
merged["_sort"] = merged["puzzle_type"].map({t: i for i, t in enumerate(TYPE_ORDER)}).fillna(len(TYPE_ORDER))
merged_sorted = merged.sort_values("_sort").drop(columns="_sort")

csv_path = "results/misses/mismatches.csv"
merged_sorted[["id", "puzzle_type", "expected", "answer"]].rename(columns={"answer": "actual"}).to_csv(csv_path, index=False)
print(f"\nSaved {len(misses)} mismatch IDs with puzzle types to {csv_path}")

print(f"\nAll type accuracies:")
for ptype in TYPE_ORDER + sorted(set(type_totals.keys()) - set(TYPE_ORDER)):
    total = type_totals.get(ptype, 0)
    matched = type_matches.get(ptype, 0)
    miss_count = total - matched
    acc = matched / total * 100 if total > 0 else 0
    print(f"  {ptype}: {matched}/{total} = {acc:.1f}% accuracy ({miss_count} misses)")

print(f"\nDone. {len(misses)} total misses across {merged['puzzle_type'].nunique()} types.")
