"""Analyze equation_numeric_deduce wrong cases.

Categorizes failures as sign_flip / prefix / suffix / wrong_num / exception.
Prints top 3 failure modes with counts and example cases.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import re
from collections import Counter, defaultdict
import pandas as pd
from src.tools import solve_equation_transform
from src.classify import classify_equation_subtype


def _same_op_examples(prompt, tgt_op):
    after = prompt.split("Below are a few examples:\n", 1)[1]
    ex_block, _ = after.split("\nNow, determine the result for: ", 1)
    out = []
    for line in ex_block.strip().splitlines():
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        lhs = lhs.strip()
        m = re.fullmatch(r"(\d+)(\D)(\d+)", lhs)
        if m and m.group(2) == tgt_op:
            out.append((m.group(1), m.group(3), rhs.strip()))
    return out


def _categorize(res, ans, tgt_op):
    res = str(res).strip()
    ans = str(ans).strip()
    if res == ans:
        return "correct"
    if res.lstrip("-") == ans.lstrip("-") and res != ans:
        return "sign_flip"
    if ans.startswith(tgt_op) and res == ans[len(tgt_op):]:
        return "missing_prefix"
    if res == tgt_op + ans:
        return "extra_prefix"
    if ans.endswith(tgt_op) and res == ans[: -len(tgt_op)]:
        return "missing_suffix"
    if res == ans + tgt_op:
        return "extra_suffix"
    try:
        r_num = int(re.sub(r"[^\d-]", "", res))
        a_num = int(re.sub(r"[^\d-]", "", ans))
        diff = r_num - a_num
        return f"wrong_num(diff={diff:+d})"
    except Exception:
        pass
    return f"other"


def run():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]
    subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == "equation_numeric_deduce")]

    cats = Counter()
    samples = defaultdict(list)  # cat -> list of dicts

    for _, r in subset.iterrows():
        ans = str(r["answer"]).strip()
        try:
            res = solve_equation_transform(json.dumps({"prompt": r["prompt"]}))
        except Exception as e:
            cats["exception"] += 1
            samples["exception"].append({"q": r["prompt"][-60:], "ans": ans, "err": str(e)[:80]})
            continue

        after = r["prompt"].split("Below are a few examples:\n", 1)[1]
        ex_block, rest = after.split("\nNow, determine the result for: ", 1)
        question = rest.strip()
        q_match = re.fullmatch(r"(\d+)(\D)(\d+)", question)
        if not q_match:
            cat = "non_standard"
        else:
            tgt_op = q_match.group(2)
            cat = _categorize(res, ans, tgt_op)

        cats[cat] += 1
        rec = {
            "q": question,
            "got": str(res).strip(),
            "exp": ans,
            "n_sameop": len(_same_op_examples(r["prompt"], q_match.group(2))) if q_match else 0,
        }
        samples[cat].append(rec)

    total = len(subset)
    correct = cats.pop("correct", 0)
    print(f"equation_numeric_deduce: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Wrong/skipped: {total - correct}")
    print()

    print("=== Failure mode breakdown ===")
    for cat, cnt in cats.most_common():
        pct = cnt / total * 100
        print(f"  {cnt:>4} ({pct:4.1f}%)  {cat}")

    print()
    print("=== Top 3 failure modes — examples ===")
    for cat, cnt in cats.most_common(3):
        print(f"\n--- {cat} ({cnt} cases) ---")
        for rec in samples[cat][:3]:
            if cat == "exception":
                print(f"  Q=...{rec['q']}  ans={rec['ans']}  err={rec['err']}")
            else:
                print(f"  Q={rec['q']}  got={rec['got']}  exp={rec['exp']}  n_sameop={rec['n_sameop']}")

    # Sign-flip deep dive
    sf = samples.get("sign_flip", [])
    if sf:
        print(f"\n=== sign_flip deep dive ({len(sf)} cases) ===")
        neg_in_training = sum(1 for r in sf if r.get("n_sameop") and
                              any(True for _ in [r]))  # placeholder
        ops = Counter(r["q"][2] for r in sf if len(r["q"]) >= 3)
        print(f"  Operator distribution: {dict(ops)}")
        print(f"  Sample: {sf[0]['q']} got={sf[0]['got']} exp={sf[0]['exp']}")


if __name__ == "__main__":
    run()
