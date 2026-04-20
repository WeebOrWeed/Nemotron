"""Analyze 164 cryptarithm_guess cases.

Finds patterns distinguishing correct from wrong predictions.
Prints recommendation: KEEP fallback or Gate on <condition>.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import pandas as pd
from collections import Counter
from src.tools import solve_equation_transform
from src.classify import classify_equation_subtype


def _parse_puzzle(prompt):
    after = prompt.split("Below are a few examples:\n", 1)[1]
    ex_block, rest = after.split("\nNow, determine the result for: ", 1)
    question = rest.strip()
    examples = []
    for line in ex_block.strip().splitlines():
        if " = " in line:
            lhs, rhs = line.split(" = ", 1)
            examples.append((lhs.strip(), rhs.strip()))
    return question, examples


def _features(question, examples, answer, res):
    tgt_op = question[2] if len(question) == 5 else None
    same_op_ex = [(lhs, rhs) for lhs, rhs in examples
                  if len(lhs) == 5 and lhs[2] == tgt_op] if tgt_op else []
    n_same_op = len(same_op_ex)
    n_total = len(examples)
    correct = (str(res).strip() == str(answer).strip())

    # result lengths
    ex_res_lens = [len(rhs) for _, rhs in same_op_ex]
    ans_len = len(str(answer).strip())
    consistent_len = len(set(ex_res_lens)) == 1 if ex_res_lens else False

    # concat pattern in same-op examples
    concat_lr = all(lhs[:2] + lhs[3:] == rhs for lhs, rhs in same_op_ex) if same_op_ex else False
    concat_rl = all(lhs[3:] + lhs[:2] == rhs for lhs, rhs in same_op_ex) if same_op_ex else False

    return {
        "correct": correct,
        "n_same_op": n_same_op,
        "n_total": n_total,
        "consistent_len": consistent_len,
        "ans_len": ans_len,
        "ex_res_len": ex_res_lens[0] if len(set(ex_res_lens)) == 1 and ex_res_lens else None,
        "concat_lr": concat_lr,
        "concat_rl": concat_rl,
    }


def run():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]
    subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == "cryptarithm_guess")]

    records = []
    for _, r in subset.iterrows():
        try:
            res = solve_equation_transform(json.dumps({"prompt": r["prompt"]}))
        except Exception as e:
            res = f"__EXCEPTION__:{e}"
        question, examples = _parse_puzzle(r["prompt"])
        feat = _features(question, examples, r["answer"], res)
        feat["predicted"] = str(res).strip()
        feat["answer"] = str(r["answer"]).strip()
        records.append(feat)

    total = len(records)
    correct = sum(1 for f in records if f["correct"])
    exceptions = sum(1 for f in records if str(f["predicted"]).startswith("__EXCEPTION__"))
    print(f"cryptarithm_guess: {correct}/{total} correct, {exceptions} exceptions")
    print()

    # Analyze correct vs wrong
    correct_recs = [f for f in records if f["correct"]]
    wrong_recs = [f for f in records if not f["correct"]]

    print("=== Correct cases breakdown ===")
    n_same_op_dist = Counter(r["n_same_op"] for r in correct_recs)
    print(f"  n_same_op dist: {dict(sorted(n_same_op_dist.items()))}")
    concat_correct = sum(1 for r in correct_recs if r["concat_lr"] or r["concat_rl"])
    print(f"  concat pattern: {concat_correct}/{len(correct_recs)}")

    print()
    print("=== Wrong cases breakdown ===")
    n_same_op_dist_w = Counter(r["n_same_op"] for r in wrong_recs)
    print(f"  n_same_op dist: {dict(sorted(n_same_op_dist_w.items()))}")
    # Are wrong cases mostly exceptions or wrong answers?
    n_exc = sum(1 for r in wrong_recs if str(r["predicted"]).startswith("__EXCEPTION__"))
    print(f"  exceptions: {n_exc}, wrong answers: {len(wrong_recs) - n_exc}")

    # Check: among wrong cases WITHOUT same-op examples, does fallback help?
    no_sameop_wrong = [r for r in wrong_recs if r["n_same_op"] == 0]
    print(f"  wrong with 0 same-op examples: {len(no_sameop_wrong)}")
    # What are we returning for these?
    predicted_lens = Counter(len(r["predicted"]) for r in no_sameop_wrong if not r["predicted"].startswith("__"))
    print(f"  predicted answer lengths: {dict(predicted_lens)}")

    print()
    # Gate analysis: would restricting to cases with same_op>0 improve?
    has_sameop = [r for r in records if r["n_same_op"] > 0]
    no_sameop = [r for r in records if r["n_same_op"] == 0]
    acc_has = sum(r["correct"] for r in has_sameop) / len(has_sameop) * 100 if has_sameop else 0
    acc_no = sum(r["correct"] for r in no_sameop) / len(no_sameop) * 100 if no_sameop else 0
    print(f"  Accuracy WITH same-op examples:    {acc_has:.1f}% ({sum(r['correct'] for r in has_sameop)}/{len(has_sameop)})")
    print(f"  Accuracy WITHOUT same-op examples: {acc_no:.1f}% ({sum(r['correct'] for r in no_sameop)}/{len(no_sameop)})")

    print()
    # Consistent result length cases
    consistent = [r for r in records if r["consistent_len"] and r["n_same_op"] > 0]
    acc_consistent = sum(r["correct"] for r in consistent) / len(consistent) * 100 if consistent else 0
    print(f"  Accuracy with consistent result length: {acc_consistent:.1f}% ({sum(r['correct'] for r in consistent)}/{len(consistent)})")

    print()
    # Recommendation
    overall_acc = correct / total * 100
    print("=== RECOMMENDATION ===")
    if acc_has > overall_acc + 5:
        print(f"Gate on n_same_op > 0  (improves from {overall_acc:.1f}% to {acc_has:.1f}%)")
    elif acc_consistent > overall_acc + 5 and len(consistent) > 20:
        print(f"Gate on consistent result length  (improves from {overall_acc:.1f}% to {acc_consistent:.1f}%)")
    else:
        print(f"KEEP fallback  (no gating condition improves by >5pp over {overall_acc:.1f}%)")


if __name__ == "__main__":
    run()
