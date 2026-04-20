"""Analyze 659 cryptarithm_deduce cases for patterns beyond concat/rev_concat.

Tests: position_map, permutation, ordinal_op, char_mapping.
Prints top 3 patterns by frequency with sample cases.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools
import re
from collections import Counter
import pandas as pd
from src.classify import classify_equation_subtype


def _parse_puzzle(prompt):
    after = prompt.split("Below are a few examples:\n", 1)[1]
    ex_block, rest = after.split("\nNow, determine the result for: ", 1)
    question = rest.strip()
    same_op = []
    all_ex = []
    if len(question) == 5:
        tgt_op = question[2]
        for line in ex_block.strip().splitlines():
            if " = " not in line:
                continue
            lhs, rhs = line.split(" = ", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            all_ex.append((lhs, rhs))
            if len(lhs) == 5 and lhs[2] == tgt_op:
                same_op.append((lhs[:2], lhs[3:], rhs))
    return question, same_op, all_ex


PRINTABLE = set(range(32, 127))


def _try_position_map(same_op, tgt_L, tgt_R, tgt_op, ans_len):
    """Try result[j] = one of [L0, L1, op, R0, R1] for each j."""
    if not same_op:
        return None
    combos = list(itertools.product(range(5), repeat=ans_len))
    for combo in combos:
        ok = True
        for L, R, res in same_op:
            if len(res) != ans_len:
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
            chars = [tgt_L[0], tgt_L[1], tgt_op, tgt_R[0], tgt_R[1]]
            return "".join(chars[p] for p in combo), combo
    return None


def _try_ordinal_op(same_op, tgt_L, tgt_R, tgt_op, ans_len):
    """Per-position: result[j] = chr((ord(inp[i]) OP ord(inp[k])) % 128)."""
    if not same_op or ans_len > 4:
        return None
    ops = [
        ("+", lambda a, b: (a + b) % 128),
        ("-", lambda a, b: (a - b) % 128),
        ("r-", lambda a, b: (b - a) % 128),
        ("^", lambda a, b: a ^ b),
        ("//2", lambda a, b: (a + b) // 2),
    ]
    per_pos = []
    for j in range(ans_len):
        found = None
        for i in range(5):
            for k in range(5):
                for op_name, op_fn in ops:
                    ok = True
                    for L, R, res in same_op:
                        if len(res) != ans_len:
                            ok = False
                            break
                        inp = [L[0], L[1], tgt_op, R[0], R[1]]
                        v = op_fn(ord(inp[i]), ord(inp[k])) % 128
                        if v not in PRINTABLE or chr(v) != res[j]:
                            ok = False
                            break
                    if ok:
                        found = (i, k, op_name, op_fn)
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None
        per_pos.append(found)
    # Apply to target
    tgt_inp = [tgt_L[0], tgt_L[1], tgt_op, tgt_R[0], tgt_R[1]]
    result = ""
    for i, k, _, op_fn in per_pos:
        v = op_fn(ord(tgt_inp[i]), ord(tgt_inp[k])) % 128
        if v not in PRINTABLE:
            return None
        result += chr(v)
    return result


def _try_char_mapping(all_ex, tgt_lhs5, ans_len):
    """Learn per-output-position char->char mapping from all examples."""
    consistent = [(lhs, rhs) for lhs, rhs in all_ex
                  if len(lhs) == 5 and len(rhs) == ans_len]
    if not consistent:
        return None
    per_pos = []
    for j in range(ans_len):
        ok = True
        seen = {}
        for lhs, rhs in consistent:
            for i in range(5):
                c_in = lhs[i]
                c_out = rhs[j]
                if c_in in seen and seen[c_in] != c_out:
                    ok = False
                    break
            if not ok:
                break
        # find which input position j is consistent
        found = None
        for i in range(5):
            mapping = {}
            valid = True
            for lhs, rhs in consistent:
                c_in = lhs[i]
                c_out = rhs[j]
                if c_in in mapping and mapping[c_in] != c_out:
                    valid = False
                    break
                mapping[c_in] = c_out
            if valid and mapping:
                found = (i, mapping)
                break
        if not found:
            return None
        per_pos.append(found)
    result = ""
    for i, mapping in per_pos:
        c = tgt_lhs5[i]
        if c not in mapping:
            return None
        result += mapping[c]
    return result


def run():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]
    subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == "cryptarithm_deduce")]

    pattern_hits = Counter()
    samples = {}  # pattern -> list of (question, answer, predicted)

    concat_correct = 0
    for _, r in subset.iterrows():
        answer = str(r["answer"]).strip()
        question, same_op, all_ex = _parse_puzzle(r["prompt"])
        if len(question) != 5:
            continue
        tgt_L, tgt_op, tgt_R = question[:2], question[2], question[3:]
        if len(tgt_L) != 2 or len(tgt_R) != 2:
            continue

        ans_len = len(answer)
        matched = False

        # Already-handled: concat
        if same_op:
            if all(L + R == res for L, R, res in same_op):
                if tgt_L + tgt_R == answer:
                    pattern_hits["concat_LR"] += 1
                    matched = True
                    concat_correct += 1
            elif all(R + L == res for L, R, res in same_op):
                if tgt_R + tgt_L == answer:
                    pattern_hits["concat_RL"] += 1
                    matched = True
                    concat_correct += 1

        if matched:
            continue

        # New: position_map (from 5-char input)
        result = _try_position_map(same_op, tgt_L, tgt_R, tgt_op, ans_len)
        if result is not None and result[0] == answer:
            pattern_hits["position_map"] += 1
            if "position_map" not in samples:
                samples["position_map"] = []
            if len(samples["position_map"]) < 3:
                samples["position_map"].append((question, answer, result[0]))
            continue

        # New: ordinal_op (per-position chr arithmetic)
        if same_op and len(same_op) >= 2:
            result2 = _try_ordinal_op(same_op, tgt_L, tgt_R, tgt_op, ans_len)
            if result2 == answer:
                pattern_hits["ordinal_op"] += 1
                if "ordinal_op" not in samples:
                    samples["ordinal_op"] = []
                if len(samples["ordinal_op"]) < 3:
                    samples["ordinal_op"].append((question, answer, result2))
                continue

        # New: char_mapping (from all examples, all ops)
        if all_ex:
            result3 = _try_char_mapping(all_ex, question, ans_len)
            if result3 == answer:
                pattern_hits["char_mapping"] += 1
                if "char_mapping" not in samples:
                    samples["char_mapping"] = []
                if len(samples["char_mapping"]) < 3:
                    samples["char_mapping"].append((question, answer, result3))
                continue

        pattern_hits["unknown"] += 1

    total = len(subset)
    print(f"cryptarithm_deduce: {total} total cases")
    print(f"concat_LR + concat_RL correct: {concat_correct}")
    print()
    print("=== Pattern frequency (top 8) ===")
    for pat, cnt in pattern_hits.most_common(8):
        pct = cnt / total * 100
        print(f"  {cnt:>5} ({pct:5.1f}%)  {pat}")

    print()
    print("=== Top 3 new patterns with samples ===")
    new_patterns = [(p, c) for p, c in pattern_hits.most_common()
                    if p not in ("concat_LR", "concat_RL", "unknown")]
    for pat, cnt in new_patterns[:3]:
        print(f"\n--- {pat} ({cnt} cases) ---")
        for question, answer, predicted in samples.get(pat, []):
            print(f"  Q={question}  answer={answer}  predicted={predicted}")

    print()
    print("=== RECOMMENDATION ===")
    new_total = sum(c for p, c in new_patterns[:3])
    if new_total >= 10:
        top_pat = new_patterns[0][0] if new_patterns else None
        print(f"Implement {top_pat} pattern (+{new_patterns[0][1]} cases if correct)")
    else:
        print(f"No new pattern ≥10 cases — KEEP current concat-only solver")
        print(f"(Top new patterns cover only {new_total} additional cases)")


if __name__ == "__main__":
    run()
