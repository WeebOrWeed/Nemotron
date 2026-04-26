"""Merge Kaggle RL artifacts into the live codebase.

Reads:
  - kaggle_output/tools_generated.py     (auto-generated tool functions)
  - kaggle_output/failure_analysis.json  (catalogue entries + metadata)
  - kaggle_output/planner_scores.jsonl   (winning DAG examples for few-shot)

Writes:
  - src/tools.py        (appends new tool defs + registers them in TOOL_REGISTRY)
  - src/planner.py      (inserts new tool catalogue entries + few-shot examples
                         into the PLANNER_SYSTEM string)

Both files are backed up to ``src/<name>.py.bak`` before modification.

Usage
-----
    python apply_artifacts.py                          # merge everything
    python apply_artifacts.py --types bit_manipulation # only bit_manipulation
    python apply_artifacts.py --dry-run                # preview without writing
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_GEN_PATH = os.path.join(HERE, "kaggle_output", "tools_generated.py")
ANALYSIS_PATH = os.path.join(HERE, "kaggle_output", "failure_analysis.json")
SCORES_PATH = os.path.join(HERE, "kaggle_output", "planner_scores.jsonl")
TOOLS_PY = os.path.join(HERE, "src", "tools.py")
PLANNER_PY = os.path.join(HERE, "src", "planner.py")

GENERATED_BLOCK_HEADER = "# === Auto-generated tools (apply_artifacts.py) ==="
GENERATED_BLOCK_FOOTER = "# === End auto-generated tools ==="
PLANNER_BLOCK_HEADER = "=== Auto-generated tools (from RL failure analysis) ==="
FEWSHOT_BLOCK_HEADER = "FEW-SHOT EXAMPLES OF CORRECT PLANS:"


# ─── Tool merging ───────────────────────────────────────────────────────

def _existing_tool_names(tools_src: str) -> set[str]:
    """Parse src/tools.py and return the set of top-level def names."""
    tree = ast.parse(tools_src)
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _strip_existing_block(src: str) -> str:
    """No-op kept for backwards compatibility.

    Earlier versions stripped the existing auto-generated block and re-wrote
    it from scratch. That was destructive when a new run produced fewer tools
    than the previous run -- the registry kept references to the old names
    while their function bodies vanished. We now simply append.
    """
    return src


def _split_generated_tools(generated_src: str) -> list[tuple[str, str]]:
    """Return [(fn_name, full_def_source), ...] for each top-level def."""
    tree = ast.parse(generated_src)
    lines = generated_src.splitlines(keepends=True)
    result: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            body = "".join(lines[start:end])
            result.append((node.name, body))
    return result


def merge_tools(allowed_types: set[str] | None, dry_run: bool) -> dict:
    """Merge generated tools into src/tools.py."""
    if not os.path.exists(TOOLS_GEN_PATH):
        print(f"  SKIP: {TOOLS_GEN_PATH} not found")
        return {"added": [], "skipped": []}

    with open(TOOLS_GEN_PATH, "r", encoding="utf-8") as f:
        generated_src = f.read()
    with open(TOOLS_PY, "r", encoding="utf-8") as f:
        tools_src = f.read()
    with open(ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    type_by_name = {t["tool_name"]: t["puzzle_type"] for t in analysis["generated_tools"]}

    cleaned_tools_src = _strip_existing_block(tools_src)
    existing_names = _existing_tool_names(cleaned_tools_src)
    all_defs = _split_generated_tools(generated_src)

    to_add: list[tuple[str, str, str]] = []  # (name, source, puzzle_type)
    skipped: list[tuple[str, str]] = []
    for name, body in all_defs:
        ptype = type_by_name.get(name, "unknown")
        if allowed_types is not None and ptype not in allowed_types:
            skipped.append((name, f"type {ptype} not in {allowed_types}"))
            continue
        if name in existing_names:
            skipped.append((name, "already exists in src/tools.py"))
            continue
        to_add.append((name, body, ptype))

    if not to_add:
        print("  No new tools to add.")
        return {"added": [], "skipped": [n for n, _ in skipped]}

    block = [f"\n\n{GENERATED_BLOCK_HEADER}\n"]
    for name, body, ptype in to_add:
        block.append(f"# [{ptype}] {name}\n{body}\n")
    block.append(f"{GENERATED_BLOCK_FOOTER}\n")
    block_text = "".join(block)

    registry_re = re.compile(
        r"(TOOL_REGISTRY:\s*dict\[str,\s*callable\]\s*=\s*\{)(.*?)(\n\})",
        re.DOTALL,
    )
    m = registry_re.search(cleaned_tools_src)
    if not m:
        sys.exit("ERROR: could not find TOOL_REGISTRY = {...} block in src/tools.py")

    new_entries = (
        "\n    # === Auto-generated (apply_artifacts.py) ===\n"
        + "".join(f'    "{name}": {name},\n' for name, _, _ in to_add)
    )
    updated_registry = m.group(1) + m.group(2).rstrip() + new_entries.rstrip("\n") + m.group(3)
    new_tools_src = (
        cleaned_tools_src[: m.start()] + updated_registry + cleaned_tools_src[m.end():]
    )
    insert_at = new_tools_src.find("TOOL_REGISTRY:")
    new_tools_src = new_tools_src[:insert_at] + block_text + "\n" + new_tools_src[insert_at:]

    print(f"  Adding {len(to_add)} new tools:")
    for name, _, ptype in to_add:
        print(f"    + [{ptype}] {name}")
    if skipped:
        print(f"  Skipping {len(skipped)} tools:")
        for name, reason in skipped:
            print(f"    - {name}  ({reason})")

    if not dry_run:
        shutil.copy(TOOLS_PY, TOOLS_PY + ".bak")
        with open(TOOLS_PY, "w", encoding="utf-8") as f:
            f.write(new_tools_src)
        print(f"  Wrote {TOOLS_PY} (backup: {TOOLS_PY}.bak)")
    else:
        print(f"  [dry-run] would write {TOOLS_PY}")

    return {"added": [n for n, _, _ in to_add], "skipped": [n for n, _ in skipped]}


# ─── Planner prompt updating ────────────────────────────────────────────

def _strip_existing_prompt_block(prompt: str) -> str:
    """Remove previously inserted catalogue + few-shot blocks."""
    prompt = re.sub(
        rf"\n*{re.escape(PLANNER_BLOCK_HEADER)}.*?(?=\n\n=== |\nRULES:|\Z)",
        "\n",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        rf"\n*{re.escape(FEWSHOT_BLOCK_HEADER)}.*?(?=\nRULES:|\Z)",
        "\n",
        prompt,
        flags=re.DOTALL,
    )
    return prompt


def _load_winners_for_fewshot(allowed_types: set[str] | None) -> dict[str, dict]:
    """Best winning example per puzzle type."""
    if not os.path.exists(SCORES_PATH):
        print(f"  No scores file at {SCORES_PATH}, skipping few-shot section.")
        return {}

    best_by_type: dict[str, dict] = {}
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("dag_valid") or r.get("reward", 0) < 1.0:
                continue
            pt = r.get("puzzle_type")
            if not pt:
                continue
            if allowed_types is not None and pt not in allowed_types:
                continue
            if pt not in best_by_type or r["reward"] > best_by_type[pt]["reward"]:
                best_by_type[pt] = r
    return best_by_type


def update_planner_prompt(allowed_types: set[str] | None, added_tools: list[str], dry_run: bool) -> None:
    with open(ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    with open(PLANNER_PY, "r", encoding="utf-8") as f:
        planner_src = f.read()

    m = re.search(
        r'(PLANNER_SYSTEM\s*=\s*"""\\?\n)(.*?)(\\?\n""")',
        planner_src,
        re.DOTALL,
    )
    if not m:
        sys.exit("ERROR: could not locate PLANNER_SYSTEM = \"\"\"...\"\"\" in src/planner.py")

    prompt_body = m.group(2)
    prompt_body = _strip_existing_prompt_block(prompt_body)

    catalogue_lines = [f"\n\n{PLANNER_BLOCK_HEADER}\n"]
    catalogue_lines.append("These tools were proposed by failure analysis and added to src/tools.py.\n")
    catalogue_lines.append("Use them when an existing per-type tool is insufficient.\n\n")
    added_set = set(added_tools)
    n_listed = 0
    for t in analysis["generated_tools"]:
        if t["tool_name"] not in added_set:
            continue
        catalogue_lines.append(
            f"- {t['tool_name']}: {t['description']}\n"
            f"  Input example: {t['catalogue_entry']}\n"
        )
        n_listed += 1
    catalogue_block = "".join(catalogue_lines) if n_listed > 0 else ""

    fewshot_records = _load_winners_for_fewshot(allowed_types)
    fewshot_lines = []
    if fewshot_records:
        fewshot_lines.append(f"\n\n{FEWSHOT_BLOCK_HEADER}\n")
        for pt, r in sorted(fewshot_records.items()):
            fewshot_lines.append(f"\n=== {pt} (reward={r['reward']}) ===\n")
            fewshot_lines.append(f"Input: PUZZLE_TYPE: {pt}\n")
            fewshot_lines.append("Output:\n")
            fewshot_lines.append(r["planner_output"][:1500].rstrip() + "\n")
    fewshot_block = "".join(fewshot_lines)

    rules_idx = prompt_body.find("\nRULES:")
    if rules_idx < 0:
        new_body = prompt_body + catalogue_block + fewshot_block
    else:
        new_body = (
            prompt_body[:rules_idx]
            + catalogue_block
            + fewshot_block
            + prompt_body[rules_idx:]
        )

    new_planner_src = planner_src[: m.start(2)] + new_body + planner_src[m.end(2):]

    print(f"  Inserting {n_listed} new tool catalogue entries")
    print(f"  Inserting {len(fewshot_records)} few-shot examples ({list(fewshot_records.keys())})")

    if not dry_run:
        shutil.copy(PLANNER_PY, PLANNER_PY + ".bak")
        with open(PLANNER_PY, "w", encoding="utf-8") as f:
            f.write(new_planner_src)
        print(f"  Wrote {PLANNER_PY} (backup: {PLANNER_PY}.bak)")
    else:
        print(f"  [dry-run] would write {PLANNER_PY}")


# ─── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Kaggle RL artifacts to local code")
    parser.add_argument("--types", nargs="*", default=None,
                        help="Restrict to specific puzzle types (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    parser.add_argument("--skip-tools", action="store_true",
                        help="Don't merge tools_generated.py into src/tools.py")
    parser.add_argument("--skip-prompt", action="store_true",
                        help="Don't update PLANNER_SYSTEM in src/planner.py")
    args = parser.parse_args()

    allowed_types = set(args.types) if args.types else None

    print("=" * 60)
    print("Apply Kaggle RL artifacts")
    print(f"  types     = {sorted(allowed_types) if allowed_types else 'ALL'}")
    print(f"  dry-run   = {args.dry_run}")
    print("=" * 60)

    added_tools: list[str] = []
    if not args.skip_tools:
        print("\n[1/2] Merging tools_generated.py -> src/tools.py")
        result = merge_tools(allowed_types, args.dry_run)
        added_tools = result["added"]
    else:
        print("\n[1/2] Tools merge skipped (--skip-tools)")

    if not args.skip_prompt:
        print("\n[2/2] Updating PLANNER_SYSTEM in src/planner.py")
        update_planner_prompt(allowed_types, added_tools, args.dry_run)
    else:
        print("\n[2/2] Prompt update skipped (--skip-prompt)")

    print("\nDone.")


if __name__ == "__main__":
    main()
