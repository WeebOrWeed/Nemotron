"""Fix helper for equation_numeric_deduce failures.

Dry-run (default): shows what would change and projected accuracy gain.
--apply: backs up src/tools.py and appends _apply_prefix_suffix_and_sign().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import re
import shutil
from collections import Counter
import pandas as pd
from src.tools import solve_equation_transform
from src.classify import classify_equation_subtype

TOOLS_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "tools.py")
BACKUP_PATH = TOOLS_PATH + ".bak_numeric_deduce"

HELPER_CODE = '''

# ── Prefix/suffix/sign post-processor (appended by 04_fix_numeric_deduce.py) ──

def _apply_prefix_suffix_and_sign(
    ans: str,
    tgt_op_ch: str,
    tgt_L_int: int,
    tgt_R_int: int,
    same_op_examples: list,
) -> str:
    """Post-process a raw numeric answer with sign/prefix/suffix rules.

    Applies:
    1. Sign correction: if training consistently shows a-b (not abs) and target
       answer would differ, preserve sign.
    2. Operator prefix: add tgt_op_ch when examples consistently show it.
    3. Trailing suffix: add common non-numeric suffix from examples.

    ``same_op_examples`` is the raw list of (L_str, R_str, result_str) for
    same-operator training cases (before any prefix/suffix stripping).
    """
    import re as _re
    if not same_op_examples:
        return ans

    # Detect and strip prefix
    def _is_op_pfx(res, op_ch):
        if not res or res[0] != op_ch:
            return False
        if op_ch == "-" and _re.fullmatch(r"\\d+", res[1:]):
            return False
        return True

    pfx_mask = [_is_op_pfx(res, tgt_op_ch) for _, _, res in same_op_examples]
    has_pfx = any(pfx_mask)

    stripped = [
        (L, R, res[1:] if _is_op_pfx(res, tgt_op_ch) else res)
        for L, R, res in same_op_examples
    ]

    # Detect trailing suffix
    trailing = {_re.sub(r"^-?\\d+", "", res) for _, _, res in stripped}
    suffix = trailing.pop() if len(trailing) == 1 and trailing != {""} else ""

    # Direction-based prefix rule
    add_prefix = False
    if has_pfx:
        r_gt_l = [int(R) > int(L) for L, R, _ in same_op_examples]
        all_pfx = all(pfx_mask)
        all_same_dir = all(r_gt_l) or not any(r_gt_l)
        if all_pfx and all_same_dir:
            dir_is_r_gt_l = r_gt_l[0]
            if dir_is_r_gt_l and tgt_R_int > tgt_L_int:
                add_prefix = True
            elif not dir_is_r_gt_l and tgt_L_int > tgt_R_int:
                add_prefix = True
        elif all_pfx:
            add_prefix = True  # mixed directions, always prefix
        else:
            consistent = all(pfx_mask[i] == r_gt_l[i] for i in range(len(same_op_examples)))
            if consistent and tgt_R_int > tgt_L_int:
                add_prefix = True

    return (tgt_op_ch if add_prefix else "") + ans + suffix
'''


def _categorize(res, ans, tgt_op):
    res, ans = str(res).strip(), str(ans).strip()
    if res == ans:
        return "correct"
    if res.lstrip("-") == ans.lstrip("-"):
        return "sign_flip"
    if ans.startswith(tgt_op) and res == ans[len(tgt_op):]:
        return "missing_prefix"
    if res == tgt_op + ans:
        return "extra_prefix"
    if ans.endswith(tgt_op) and res == ans[: -len(tgt_op)]:
        return "missing_suffix"
    if res == ans + tgt_op:
        return "extra_suffix"
    return "other"


def audit():
    df = pd.read_csv("data/train.csv")
    eq = df[df["prompt"].str.contains("transformation rules is applied to equations")]
    subset = eq[eq["prompt"].apply(lambda p: classify_equation_subtype(p) == "equation_numeric_deduce")]

    cats = Counter()
    for _, r in subset.iterrows():
        ans = str(r["answer"]).strip()
        try:
            res = solve_equation_transform(json.dumps({"prompt": r["prompt"]}))
        except Exception:
            cats["exception"] += 1
            continue
        after = r["prompt"].split("Below are a few examples:\n", 1)[1]
        _, rest = after.split("\nNow, determine the result for: ", 1)
        q_match = re.fullmatch(r"(\d+)(\D)(\d+)", rest.strip())
        cat = _categorize(res, ans, q_match.group(2)) if q_match else "non_standard"
        cats[cat] += 1

    total = len(subset)
    correct = cats.pop("correct", 0)
    print(f"Current: {correct}/{total} = {correct/total*100:.1f}%")
    print("Failures:", dict(cats.most_common()))

    fixable = cats.get("missing_prefix", 0) + cats.get("extra_prefix", 0) + cats.get("missing_suffix", 0)
    print(f"\nEstimated fixable (prefix/suffix): {fixable} cases")
    print(f"Projected after fix: ~{correct + fixable}/{total} = ~{(correct + fixable)/total*100:.1f}%")
    return correct, total, fixable


def apply_fix():
    if os.path.exists(BACKUP_PATH):
        print(f"Backup already exists: {BACKUP_PATH}")
    else:
        shutil.copy2(TOOLS_PATH, BACKUP_PATH)
        print(f"Backed up to {BACKUP_PATH}")

    with open(TOOLS_PATH, "a") as f:
        f.write(HELPER_CODE)
    print(f"Appended _apply_prefix_suffix_and_sign() to {TOOLS_PATH}")
    print("NOTE: You must manually wire this helper into solve_equation_transform().")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write the fix to src/tools.py")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    correct, total, fixable = audit()
    if args.apply:
        print()
        apply_fix()
    else:
        print("\n(dry run — pass --apply to write the fix)")


if __name__ == "__main__":
    main()
