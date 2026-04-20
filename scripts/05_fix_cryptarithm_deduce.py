"""Fix helper for cryptarithm_deduce: add position_map pattern.

Dry-run (default): shows how many cases position_map covers and accuracy gain.
--apply: backs up src/tools.py and appends _try_position_map_cryptarithm().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import itertools
import json
import re
import shutil
from collections import Counter
import pandas as pd
from src.classify import classify_equation_subtype

TOOLS_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "tools.py")
BACKUP_PATH = TOOLS_PATH + ".bak_crypto_deduce"

HELPER_CODE = '''

# ── Position-map solver for cryptarithm_deduce (appended by 05_fix_cryptarithm_deduce.py) ──

def _try_position_map_cryptarithm(
    same_op: list,
    tgt_L: str,
    tgt_op: str,
    tgt_R: str,
    ans_len: int,
) -> str | None:
    """Try result[j] = one of [L0, L1, op, R0, R1] for each output position j.

    same_op: list of (L_str, R_str, result_str) — same operator as target.
    Returns predicted result string or None if no consistent mapping found.
    Requires ≥2 examples for a reliable match (≥1 example for 1-char result).
    """
    import itertools as _it
    if not same_op:
        return None
    min_examples = 1 if ans_len == 1 else 2
    if len(same_op) < min_examples:
        return None

    combos = list(_it.product(range(5), repeat=ans_len))
    for combo in combos:
        ok = True
        for L, R, res in same_op:
            if len(L) != 2 or len(R) != 2 or len(res) != ans_len:
                ok = False
                break
            chars = [L[0], L[1], tgt_op, R[0], R[1]]
            for j, pos in enumerate(combo):
                if chars[pos] != res[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            tgt_chars = [tgt_L[0], tgt_L[1], tgt_op, tgt_R[0], tgt_R[1]]
            return "".join(tgt_chars[p] for p in combo)
    return None
'''


def _parse_puzzle(prompt):
    after = prompt.split("Below are a few examples:\n", 1)[1]
    ex_block, rest = after.split("\nNow, determine the result for: ", 1)
    question = rest.strip()
    same_op = []
    if len(question) == 5:
        tgt_op = question[2]
        for line in ex_block.strip().splitlines():
            if " = " not in line:
                continue
            lhs, rhs = line.split(" = ", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            if len(lhs) == 5 and lhs[2] == tgt_op:
                same_op.append((lhs[:2], lhs[3:], rhs))
    return question, same_op


def _try_position_map(same_op, tgt_L, tgt_op, tgt_R, ans_len):
    if not same_op or len(same_op) < (1 if ans_len == 1 else 2):
        return None
    combos = list(itertools.product(range(5), repeat=ans_len))
    for combo in combos:
        ok = True
        for L, R, res in same_op:
            if len(L) != 2 or len(R) != 2 or len(res) != ans_len:
                ok = False
                break
            chars = [L[0], L[1], tgt_op, R[0], R[1]]
            for j, pos in enumerate(combo):
                if chars[pos] != res[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            tgt_chars = [tgt_L[0], tgt_L[1], tgt_op, tgt_R[0], tgt_R[1]]
            return "".join(tgt_chars[p] for p in combo)
    return None


def audit():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]
    subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == "cryptarithm_deduce")]

    from src.tools import solve_equation_transform
    current_correct = 0
    position_map_new = 0
    samples = []

    for _, r in subset.iterrows():
        answer = str(r["answer"]).strip()
        question, same_op = _parse_puzzle(r["prompt"])
        if len(question) != 5:
            continue
        tgt_L, tgt_op, tgt_R = question[:2], question[2], question[3:]
        if len(tgt_L) != 2 or len(tgt_R) != 2:
            continue

        # Current solver
        try:
            res = solve_equation_transform(json.dumps({"prompt": r["prompt"]}))
            currently_correct = str(res).strip() == answer
        except Exception:
            currently_correct = False

        if currently_correct:
            current_correct += 1
            continue

        # Position map
        pred = _try_position_map(same_op, tgt_L, tgt_op, tgt_R, len(answer))
        if pred == answer:
            position_map_new += 1
            if len(samples) < 5:
                samples.append((question, answer, pred, same_op[:2]))

    total = len(subset)
    print(f"cryptarithm_deduce: {current_correct}/{total} currently correct ({current_correct/total*100:.1f}%)")
    print(f"position_map would add: {position_map_new} new correct cases")
    print(f"Projected: {current_correct + position_map_new}/{total} = {(current_correct + position_map_new)/total*100:.1f}%")
    print()
    print("Sample position_map cases:")
    for q, ans, pred, ex in samples:
        print(f"  Q={q}  answer={ans}")
        for L, R, res in ex:
            print(f"    {L}?{R} = {res}")
    return current_correct, total, position_map_new


def apply_fix():
    if os.path.exists(BACKUP_PATH):
        print(f"Backup already exists: {BACKUP_PATH}")
    else:
        shutil.copy2(TOOLS_PATH, BACKUP_PATH)
        print(f"Backed up to {BACKUP_PATH}")

    with open(TOOLS_PATH, "a") as f:
        f.write(HELPER_CODE)
    print(f"Appended _try_position_map_cryptarithm() to {TOOLS_PATH}")
    print("NOTE: Wire this into solve_equation_transform() in the cryptarithm_deduce block.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write the fix to src/tools.py")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    current_correct, total, new_cases = audit()
    if args.apply:
        print()
        apply_fix()
    else:
        print("\n(dry run — pass --apply to write the fix)")
        if new_cases < 10:
            print(f"WARNING: only {new_cases} new cases — threshold is 10, skip --apply")


if __name__ == "__main__":
    main()
