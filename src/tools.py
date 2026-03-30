"""Deterministic tool functions for DAG node execution.

Each tool takes a JSON-parseable `tool_input` string and returns a string result.
These bypass the LLM entirely for computation -- pure Python, 100% accurate.

The special tool "ask_llm" is handled by the solver (routes to the LLM).
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re


# ===================================================================
# Safe math evaluator
# ===================================================================

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs, "round": round, "int": int, "float": float,
    "min": min, "max": max, "sum": sum, "len": len,
    "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
    "log": math.log, "log10": math.log10, "pow": math.pow,
}


def _safe_eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return op(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unknown function: {node.func.id}")
        args = [_safe_eval_node(a) for a in node.args]
        return func(*args)
    if isinstance(node, ast.List):
        return [_safe_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(e) for e in node.elts)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _fmt_number(result) -> str:
    if isinstance(result, float):
        if result == int(result) and abs(result) < 1e15:
            return str(int(result))
    return str(result)


# ===================================================================
# General tools
# ===================================================================

def eval_math(tool_input: str) -> str:
    """Safely evaluate a math expression.
    Input: {"expr": "0.5 * 9.8 * 3**2"}
    """
    data = json.loads(tool_input)
    tree = ast.parse(data["expr"], mode="eval")
    return _fmt_number(_safe_eval_node(tree.body))


def apply_formula(tool_input: str) -> str:
    """Evaluate a formula with variable substitution.
    Input: {"formula": "0.5 * g * t ** 2", "vars": {"g": 9.8, "t": 3}}
    """
    data = json.loads(tool_input)
    formula = data["formula"]
    for name, value in data.get("vars", {}).items():
        formula = re.sub(rf"\b{re.escape(name)}\b", str(value), formula)
    tree = ast.parse(formula, mode="eval")
    return _fmt_number(_safe_eval_node(tree.body))


def round_number(tool_input: str) -> str:
    """Round a number to n decimal places.
    Input: {"value": 154.6234, "decimals": 2}
    """
    data = json.loads(tool_input)
    val = float(data["value"])
    dec = int(data.get("decimals", 2))
    return f"{val:.{dec}f}"


def average(tool_input: str) -> str:
    """Compute the average of a list of numbers.
    Input: {"values": [15.88, 15.92, 15.87]}
    """
    data = json.loads(tool_input)
    vals = [float(v) for v in data["values"]]
    return _fmt_number(sum(vals) / len(vals))


def regex_extract(tool_input: str) -> str:
    """Extract all regex matches from text. Returns JSON array.
    Input: {"text": "t = 1.37s, distance = 14.92 m", "pattern": "[\\d.]+"}
    """
    data = json.loads(tool_input)
    matches = re.findall(data["pattern"], data["text"])
    return json.dumps(matches)


# ===================================================================
# Bit manipulation tools (8-bit binary)
# ===================================================================

def xor_binary(tool_input: str) -> str:
    """XOR two binary strings.
    Input: {"a": "10110010", "b": "01001101"}
    """
    data = json.loads(tool_input)
    a, b = data["a"].strip(), data["b"].strip()
    width = max(len(a), len(b))
    return format(int(a, 2) ^ int(b, 2), f"0{width}b")


def and_binary(tool_input: str) -> str:
    """AND two binary strings.
    Input: {"a": "10110010", "b": "01001101"}
    """
    data = json.loads(tool_input)
    a, b = data["a"].strip(), data["b"].strip()
    width = max(len(a), len(b))
    return format(int(a, 2) & int(b, 2), f"0{width}b")


def or_binary(tool_input: str) -> str:
    """OR two binary strings.
    Input: {"a": "10110010", "b": "01001101"}
    """
    data = json.loads(tool_input)
    a, b = data["a"].strip(), data["b"].strip()
    width = max(len(a), len(b))
    return format(int(a, 2) | int(b, 2), f"0{width}b")


def not_binary(tool_input: str) -> str:
    """Flip all bits in a binary string.
    Input: {"a": "10110010"} or {"a": "10110010", "bits": 8}
    """
    data = json.loads(tool_input)
    a = data["a"].strip()
    bits = int(data.get("bits", len(a)))
    mask = (1 << bits) - 1
    return format(int(a, 2) ^ mask, f"0{bits}b")


def shift_left(tool_input: str) -> str:
    """Left-shift a binary string by n positions (zero-fill).
    Input: {"a": "10110010", "n": 1, "bits": 8}
    """
    data = json.loads(tool_input)
    a = data["a"].strip()
    n, bits = int(data["n"]), int(data.get("bits", len(a)))
    return format((int(a, 2) << n) & ((1 << bits) - 1), f"0{bits}b")


def shift_right(tool_input: str) -> str:
    """Right-shift a binary string by n positions (zero-fill).
    Input: {"a": "10110010", "n": 1, "bits": 8}
    """
    data = json.loads(tool_input)
    a = data["a"].strip()
    n, bits = int(data["n"]), int(data.get("bits", len(a)))
    return format(int(a, 2) >> n, f"0{bits}b")


def rotate_left(tool_input: str) -> str:
    """Circular rotate-left a binary string.
    Input: {"a": "10110010", "n": 1, "bits": 8}
    """
    data = json.loads(tool_input)
    a = data["a"].strip()
    n, bits = int(data["n"]), int(data.get("bits", len(a)))
    val = int(a, 2)
    n = n % bits
    result = ((val << n) | (val >> (bits - n))) & ((1 << bits) - 1)
    return format(result, f"0{bits}b")


def rotate_right(tool_input: str) -> str:
    """Circular rotate-right a binary string.
    Input: {"a": "10110010", "n": 1, "bits": 8}
    """
    data = json.loads(tool_input)
    a = data["a"].strip()
    n, bits = int(data["n"]), int(data.get("bits", len(a)))
    val = int(a, 2)
    n = n % bits
    result = ((val >> n) | (val << (bits - n))) & ((1 << bits) - 1)
    return format(result, f"0{bits}b")


# ===================================================================
# Cipher / substitution tools
# ===================================================================

def substitute_chars(tool_input: str) -> str:
    """Apply a character substitution mapping to text.
    Input: {"text": "hello", "mapping": {"h": "x", "e": "y", "l": "z", "o": "w"}}
    """
    data = json.loads(tool_input)
    text = data["text"]
    mapping = data["mapping"]
    return "".join(mapping.get(ch, ch) for ch in text)


def build_char_map(tool_input: str) -> str:
    """Build a character substitution map from aligned text pairs.
    Input: {"pairs": [["ucoov", "queen"], ["pqrsfv", "dragon"]]}
    Returns JSON object of the mapping.
    """
    data = json.loads(tool_input)
    pairs = data["pairs"]
    if isinstance(pairs, str):
        pairs = json.loads(pairs)
    mapping: dict[str, str] = {}
    for encrypted, plain in pairs:
        for e_ch, p_ch in zip(encrypted, plain):
            if e_ch != " " and p_ch != " ":
                if e_ch in mapping and mapping[e_ch] != p_ch:
                    pass  # conflict -- keep first seen
                else:
                    mapping[e_ch] = p_ch
    return json.dumps(mapping, ensure_ascii=False)


# ===================================================================
# Numeral conversion tools (dataset is 100% Roman numerals)
# ===================================================================

_ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

_ROMAN_MAP = {
    "I": 1, "V": 5, "X": 10, "L": 50,
    "C": 100, "D": 500, "M": 1000,
}


def to_roman(tool_input: str) -> str:
    """Convert an integer to a Roman numeral string.
    Input: {"number": 38}
    """
    data = json.loads(tool_input)
    raw = data["number"]
    # Tolerate string inputs like "38" or floats like 38.0
    num = int(float(str(raw).strip()))
    if num <= 0 or num > 3999:
        raise ValueError(f"Roman numerals support 1-3999, got {num}")
    result = []
    for value, numeral in _ROMAN_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)


def from_roman(tool_input: str) -> str:
    """Convert a Roman numeral string to an integer.
    Input: {"roman": "XXXVIII"}
    """
    data = json.loads(tool_input)
    roman = data["roman"].strip().upper()
    total = 0
    prev = 0
    for ch in reversed(roman):
        val = _ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return str(total)


# ===================================================================
# Unit conversion / gravity tools
# ===================================================================

def _to_float_pairs(pairs) -> list[list[float]]:
    """Coerce pairs to [[float, float], ...], tolerant of string values."""
    result = []
    if isinstance(pairs, str):
        pairs = json.loads(pairs)
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            result.append([float(pair[0]), float(pair[1])])
    return result


def linear_factor(tool_input: str) -> str:
    """Compute average linear conversion factor from (input, output) pairs.
    Input: {"pairs": [[10.08, 6.69], [17.83, 11.83], [35.85, 23.79]]}
    """
    data = json.loads(tool_input)
    pairs = _to_float_pairs(data["pairs"])
    factors = [y / x for x, y in pairs if x != 0]
    if not factors:
        raise ValueError("All x-values are zero")
    avg = sum(factors) / len(factors)
    return _fmt_number(avg)


def compute_gravity_g(tool_input: str) -> str:
    """Compute g from (t, d) observation pairs using g = 2d/t^2. Returns average g.
    Input: {"observations": [[1.37, 14.92], [4.27, 144.96], [3.28, 85.54]]}
    Each pair is [t_seconds, d_meters].
    """
    data = json.loads(tool_input)
    obs = _to_float_pairs(data["observations"])
    g_values = [2 * d / (t ** 2) for t, d in obs if t != 0]
    if not g_values:
        raise ValueError("No valid observations")
    avg_g = sum(g_values) / len(g_values)
    return _fmt_number(avg_g)


def compute_gravity_d(tool_input: str) -> str:
    """Compute falling distance d = 0.5 * g * t^2, rounded to 2 decimal places.
    Input: {"g": 15.89, "t": 4.41}
    """
    data = json.loads(tool_input)
    g = float(data["g"])
    t = float(data["t"])
    d = 0.5 * g * t ** 2
    return f"{d:.2f}"


# ===================================================================
# End-to-end solvers (parse prompt with regex, compute, return answer)
# ===================================================================

def solve_gravity_physics(tool_input: str) -> str:
    """Solve a gravity puzzle end-to-end from the raw prompt."""
    data = json.loads(tool_input)
    prompt = data["prompt"]
    obs_matches = re.findall(
        r't\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)\s*m', prompt
    )
    observations = [[float(t), float(d)] for t, d in obs_matches]
    target_match = re.search(
        r'for\s+t\s*=\s*([\d.]+)\s*s?\s*given', prompt, re.IGNORECASE
    )
    if not target_match:
        target_match = re.search(r'for\s+t\s*=\s*([\d.]+)', prompt, re.IGNORECASE)
    target_t = float(target_match.group(1))
    g_values = [2 * d / (t ** 2) for t, d in observations if t != 0]
    avg_g = sum(g_values) / len(g_values)
    d = 0.5 * avg_g * target_t ** 2
    return f"{d:.2f}"


def solve_unit_conversion(tool_input: str) -> str:
    """Solve a unit conversion puzzle end-to-end from the raw prompt."""
    data = json.loads(tool_input)
    prompt = data["prompt"]
    pairs = re.findall(r'([\d.]+)\s*m?\s*becomes\s*([\d.]+)', prompt)
    float_pairs = [[float(x), float(y)] for x, y in pairs]
    factors = [y / x for x, y in float_pairs if x != 0]
    avg_factor = sum(factors) / len(factors)
    target_match = re.search(
        r'convert.*?(?:measurement)?[:\s]+([\d.]+)\s*m', prompt, re.IGNORECASE
    )
    target = float(target_match.group(1))
    result = avg_factor * target
    return f"{result:.2f}"


def solve_numeral_conversion(tool_input: str) -> str:
    """Solve a numeral conversion puzzle (always Roman numerals)."""
    data = json.loads(tool_input)
    prompt = data["prompt"]
    match = re.search(r'the number\s+(\d+)', prompt, re.IGNORECASE)
    num = int(match.group(1))
    result = []
    for value, numeral in _ROMAN_VALUES:
        while num >= value:
            result.append(numeral)
            num -= value
    return "".join(result)


_CIPHER_VOCAB = {
    "above", "alice", "ancient", "around", "beyond", "bird", "book", "bright",
    "castle", "cat", "cave", "chases", "clever", "colorful", "creates",
    "crystal", "curious", "dark", "discovers", "door", "dragon", "draws",
    "dreams", "explores", "follows", "forest", "found", "garden", "golden",
    "hatter", "hidden", "imagines", "in", "inside", "island", "key", "king",
    "knight", "library", "magical", "map", "message", "mirror", "mountain",
    "mouse", "mysterious", "near", "ocean", "palace", "potion", "princess",
    "puzzle", "queen", "rabbit", "reads", "school", "secret", "sees", "silver",
    "story", "strange", "student", "studies", "teacher", "the", "through",
    "tower", "treasure", "turtle", "under", "valley", "village", "watches",
    "wise", "wizard", "wonderland", "writes",
}


def solve_cipher_decryption(tool_input: str) -> str:
    """Solve a cipher puzzle by extracting word pairs and building a substitution map."""
    from itertools import permutations as _perms

    data = json.loads(tool_input)
    prompt = data["prompt"]
    example_lines = re.findall(r'(.+?)\s*->\s*(.+)', prompt)
    mapping: dict[str, str] = {}
    for enc_line, plain_line in example_lines:
        enc_words = enc_line.strip().split()
        plain_words = plain_line.strip().split()
        for enc_w, plain_w in zip(enc_words, plain_words):
            for e_ch, p_ch in zip(enc_w, plain_w):
                if e_ch not in mapping:
                    mapping[e_ch] = p_ch

    target_match = re.search(
        r'decrypt the following text:\s*(.+)', prompt, re.IGNORECASE
    )
    target = target_match.group(1).strip()

    # Find unmapped letters needed for the target
    all_letters = set("abcdefghijklmnopqrstuvwxyz")
    mapped_to = set(mapping.values()) & all_letters
    unmapped_in_target = sorted({ch for ch in target if ch in all_letters and ch not in mapping})
    unmapped_plain = sorted(all_letters - mapped_to)

    if unmapped_in_target and len(unmapped_plain) >= len(unmapped_in_target):
        # Try permutations of just the letters we actually need for the target
        best_result = None
        best_score = -1
        for perm in _perms(unmapped_plain, len(unmapped_in_target)):
            test_map = dict(mapping)
            for e, p in zip(unmapped_in_target, perm):
                test_map[e] = p
            result = "".join(test_map.get(ch, ch) for ch in target)
            words = result.split()
            score = sum(1 for w in words if w in _CIPHER_VOCAB)
            if score > best_score:
                best_score = score
                best_result = result
            if score == len(words):
                return result
        if best_result:
            return best_result

    return "".join(mapping.get(ch, ch) for ch in target)


def _choice(a: int, b: int, c: int) -> int:
    return (a & b) | ((1 - a) & c)


def _majority(a: int, b: int, c: int) -> int:
    return (a & b) | (a & c) | (b & c)


def _brute_force_bit_rule(
    inputs: list[int], outputs: list[int], target: int
) -> int | None:
    """Find the bit manipulation rule using per-bit boolean function search.

    Strategy:
    1. Per-bit: 1-input, then 2-input, then 3-input truth-table matching (unanimous)
    2. Find global sliding-window choice/majority pattern; override if coverage >= 4
    3. For still-undetermined bits, use any matching choice/majority
    4. For still-undetermined bits, try 4-input and 5-input with majority vote
    """
    from itertools import combinations as _combs

    BITS = 8
    n_ex = len(inputs)
    input_cols = [tuple((i >> b) & 1 for i in inputs) for b in range(BITS)]
    target_bits = [(target >> b) & 1 for b in range(BITS)]
    out_cols = [tuple((o >> b) & 1 for o in outputs) for b in range(BITS)]
    result_bits = [None] * BITS

    def _try_n_input(bit_positions, out_col: tuple) -> int | None:
        cols = [input_cols[p] for p in bit_positions]
        t_vals = [target_bits[p] for p in bit_positions]
        observed: dict[tuple, int] = {}
        for k in range(n_ex):
            key = tuple(c[k] for c in cols)
            if key in observed:
                if observed[key] != out_col[k]:
                    return None
            else:
                observed[key] = out_col[k]
        target_key = tuple(t_vals)
        return observed.get(target_key)

    # Phase 1: full per-bit search (1-input, 2-input, 3-input)
    for out_pos in range(BITS):
        oc = out_cols[out_pos]
        if all(b == 0 for b in oc):
            result_bits[out_pos] = 0
            continue
        if all(b == 1 for b in oc):
            result_bits[out_pos] = 1
            continue

        for a in range(BITS):
            val = _try_n_input([a], oc)
            if val is not None:
                result_bits[out_pos] = val
                break
        if result_bits[out_pos] is not None:
            continue

        vals_2: set[int] = set()
        for a in range(BITS):
            for b in range(a, BITS):
                val = _try_n_input([a, b], oc)
                if val is not None:
                    vals_2.add(val)
        if len(vals_2) == 1:
            result_bits[out_pos] = vals_2.pop()
            continue

        vals_3: set[int] = set()
        for a in range(BITS):
            for b in range(a + 1, BITS):
                for c in range(b + 1, BITS):
                    val = _try_n_input([a, b, c], oc)
                    if val is not None:
                        vals_3.add(val)
        if len(vals_3) == 1:
            result_bits[out_pos] = vals_3.pop()

    # Phase 2: global sliding-window choice/majority pattern
    best_pattern: dict[int, int] | None = None
    best_coverage = 0
    for oa in range(BITS):
        for ob in range(BITS):
            for oc in range(BITS):
                if oa == ob == oc:
                    continue
                for func in [_choice, _majority]:
                    pvals: dict[int, int] = {}
                    coverage = 0
                    for out_pos in range(BITS):
                        a = (out_pos + oa) % BITS
                        b = (out_pos + ob) % BITS
                        c = (out_pos + oc) % BITS
                        computed = tuple(
                            func(input_cols[a][k], input_cols[b][k], input_cols[c][k])
                            for k in range(n_ex)
                        )
                        if computed == out_cols[out_pos]:
                            pvals[out_pos] = func(target_bits[a], target_bits[b], target_bits[c])
                            coverage += 1
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_pattern = dict(pvals)

    if best_pattern and best_coverage >= 4:
        for pos, val in best_pattern.items():
            result_bits[pos] = val

    # Phase 3: cascading majority vote (prefer lowest arity that has a winner)
    for out_pos in range(BITS):
        if result_bits[out_pos] is not None:
            continue
        oc = out_cols[out_pos]
        for arity in (2, 3, 4, 5):
            votes = {0: 0, 1: 0}
            for combo in _combs(range(BITS), arity):
                val = _try_n_input(combo, oc)
                if val is not None:
                    votes[val] += 1
            total = votes[0] + votes[1]
            if total > 0 and votes[0] != votes[1]:
                result_bits[out_pos] = 1 if votes[1] > votes[0] else 0
                break
        else:
            # All arities tied or empty; fall back to cumulative vote
            votes = {0: 0, 1: 0}
            for arity in (2, 3, 4, 5):
                for combo in _combs(range(BITS), arity):
                    val = _try_n_input(combo, oc)
                    if val is not None:
                        votes[val] += 1
            if votes[0] + votes[1] > 0:
                result_bits[out_pos] = 1 if votes[1] >= votes[0] else 0

    if any(b is None for b in result_bits):
        return None
    return sum(b << i for i, b in enumerate(result_bits))


def solve_bit_manipulation(tool_input: str) -> str:
    """Solve a bit manipulation puzzle by brute-forcing the operation."""
    data = json.loads(tool_input)
    prompt = data["prompt"]
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not pairs:
        raise ValueError("No binary pairs found in prompt")
    inputs = [int(a, 2) for a, _ in pairs]
    outputs = [int(b, 2) for _, b in pairs]
    target_match = re.search(
        r'(?:determine|find|compute).*?:\s*([01]{8})', prompt, re.IGNORECASE
    )
    if not target_match:
        target_match = re.search(r'for:?\s*([01]{8})', prompt, re.IGNORECASE)
    target = int(target_match.group(1), 2)
    result = _brute_force_bit_rule(inputs, outputs, target)
    if result is not None:
        return format(result, '08b')
    raise ValueError("Could not find consistent bit manipulation rule")


def solve_equation_transform(tool_input: str) -> str:
    """Solve equation_transform using the operator-centric compound-operation framework.

    Framework:
    - Center char (position 2) = operator (defines a compound operation)
    - Left 2 chars + Right 2 chars = operands
    - The operator may be a compound of primitives: +, -, *, /, reverse, abs, etc.
    - Tries numeric strategies (when chars are digits) and symbolic (ordinal-based).
    """
    data = json.loads(tool_input)
    prompt = data["prompt"]

    lines = prompt.strip().split("\n")
    examples: list[tuple[str, str]] = []
    target = None
    for line in lines:
        line = line.strip()
        if " = " in line and "Now" not in line and "Alice" not in line and "secret" not in line:
            parts = line.split(" = ", 1)
            if len(parts) == 2 and len(parts[0]) == 5:
                examples.append((parts[0], parts[1]))
        if "determine the result for:" in line:
            target = line.split("determine the result for:")[-1].strip()

    if not examples or not target:
        raise ValueError("Could not parse equation_transform puzzle")
    if len(target) != 5:
        raise ValueError("Non-standard equation format")

    target_op = target[2]
    t_left, t_right = target[:2], target[3:]

    by_op: dict[str, list[tuple[str, str, str]]] = {}
    for lhs, rhs in examples:
        op = lhs[2]
        by_op.setdefault(op, []).append((lhs[:2], lhs[3:], rhs))

    same_op = by_op.get(target_op, [])
    all_parsed = [(lhs[:2], lhs[2], lhs[3:], rhs) for lhs, rhs in examples]

    # --- Helper: reverse a number's digits ---
    def _rev_num(n: int) -> int:
        s = str(abs(n))[::-1].lstrip("0") or "0"
        return int(s) * (1 if n >= 0 else -1)

    # --- Check if all operand chars (excluding operators) are digits ---
    operand_chars = "".join(l + r for l, _, r, _ in all_parsed) + t_left + t_right
    is_numeric_puzzle = all(c.isdigit() for c in operand_chars)

    # =================================================================
    # STRATEGY 1: Concatenation (left + right, works for both modes)
    # =================================================================
    if same_op and all(l + r == res for l, r, res in same_op):
        return t_left + t_right
    if all(l + r == res for l, _, r, res in all_parsed):
        return t_left + t_right
    if same_op and all(r + l == res for l, r, res in same_op):
        return t_right + t_left

    # =================================================================
    # STRATEGY 2: Numeric compound operations
    # =================================================================
    if is_numeric_puzzle and same_op:
        binary_ops = [
            ("L+R", lambda l, r: l + r),
            ("L-R", lambda l, r: l - r),
            ("R-L", lambda l, r: r - l),
            ("L*R", lambda l, r: l * r),
            ("L*R-1", lambda l, r: l * r - 1),
            ("L*R+1", lambda l, r: l * r + 1),
            ("abs_diff", lambda l, r: abs(l - r)),
        ]
        if all(int(r) != 0 for _, r, _ in same_op):
            binary_ops.append(("L//R", lambda l, r: l // r if r else None))
            binary_ops.append(("L%R", lambda l, r: l % r if r else None))
        if all(int(l) != 0 for l, _, _ in same_op):
            binary_ops.append(("R//L", lambda l, r: r // l if l else None))
            binary_ops.append(("R%L", lambda l, r: r % l if l else None))
        binary_ops += [
            ("gcd", lambda l, r: math.gcd(l, r) if l and r else 0),
            ("max", lambda l, r: max(l, r)),
            ("min", lambda l, r: min(l, r)),
            ("xor", lambda l, r: l ^ r),
            ("or", lambda l, r: l | r),
            ("and", lambda l, r: l & r),
        ]

        def _rev_str(s: str) -> int:
            """Reverse a 2-digit string then parse: '02' -> '20' -> 20."""
            return int(s[::-1])

        pre_transforms = [
            ("rev_both", lambda ls, rs: (_rev_str(ls), _rev_str(rs))),
            ("as_is", lambda ls, rs: (int(ls), int(rs))),
            ("swap", lambda ls, rs: (int(rs), int(ls))),
            ("rev_L", lambda ls, rs: (_rev_str(ls), int(rs))),
            ("rev_R", lambda ls, rs: (int(ls), _rev_str(rs))),
        ]

        post_transforms = [
            ("rev", lambda x: str(abs(x))[::-1].lstrip("0") or "0"),
            ("neg_rev", lambda x: ("-" if x < 0 else "") + (str(abs(x))[::-1].lstrip("0") or "0")),
            ("id", str),
            ("abs", lambda x: str(abs(x))),
        ]

        first_match = None
        for _, pre_fn in pre_transforms:
            for _, op_fn in binary_ops:
                for _, post_fn in post_transforms:
                    ok = True
                    for l_s, r_s, res in same_op:
                        pl, pr = pre_fn(l_s, r_s)
                        v = op_fn(pl, pr)
                        if v is None or post_fn(v) != res:
                            ok = False
                            break
                    if ok:
                        pl, pr = pre_fn(t_left, t_right)
                        v = op_fn(pl, pr)
                        if v is not None:
                            pred = post_fn(v)
                            if len(same_op) >= 2:
                                return pred
                            if first_match is None:
                                first_match = pred

        if first_match is not None:
            return first_match

        # Numeric concatenation: L_str + R_str as strings
        if all(l + r == res for l, r, res in same_op):
            return t_left + t_right
        if all(r + l == res for l, r, res in same_op):
            return t_right + t_left

    # =================================================================
    # STRATEGY 3: Numeric with ALL examples (cross-operator)
    # =================================================================
    if is_numeric_puzzle and not same_op:
        all_ex = [(l, r, res) for l, _, r, res in all_parsed]
        simple_ops = [
            lambda l, r: l + r,
            lambda l, r: l - r,
            lambda l, r: r - l,
            lambda l, r: l * r,
            lambda l, r: l * r - 1,
            lambda l, r: abs(l - r),
        ]
        for op_fn in simple_ops:
            ok = True
            for l_s, r_s, res in all_ex:
                v = op_fn(int(l_s), int(r_s))
                if str(v) != res:
                    ok = False
                    break
            if ok:
                return str(op_fn(int(t_left), int(t_right)))

    # =================================================================
    # STRATEGY 4: Symbolic per-char ordinal operations (2-char output)
    # =================================================================
    if same_op:
        two_char = [(l, r, res) for l, r, res in same_op if len(res) == 2]
        if two_char and len(two_char) >= 2:
            per_char_ops = [
                lambda a, b: a + b,
                lambda a, b: a - b,
                lambda a, b: b - a,
                lambda a, b: a ^ b,
                lambda a, b: a | b,
                lambda a, b: a & b,
                lambda a, b: max(a, b),
                lambda a, b: min(a, b),
                lambda a, b: abs(a - b),
                lambda a, b: a * b,
            ]
            for op_fn in per_char_ops:
                for M in [94, 95, 127, 128, 256]:
                    for off in [0, 33, 32, -33]:
                        ok = True
                        for l, r, res in two_char:
                            if len(res) != 2:
                                ok = False
                                break
                            try:
                                v0 = (op_fn(ord(l[0]), ord(r[0])) + off) % M
                                v1 = (op_fn(ord(l[1]), ord(r[1])) + off) % M
                                if chr(v0) + chr(v1) != res:
                                    ok = False
                                    break
                            except (ValueError, ZeroDivisionError):
                                ok = False
                                break
                        if ok:
                            try:
                                v0 = (op_fn(ord(t_left[0]), ord(t_right[0])) + off) % M
                                v1 = (op_fn(ord(t_left[1]), ord(t_right[1])) + off) % M
                                return chr(v0) + chr(v1)
                            except (ValueError, ZeroDivisionError):
                                pass

    # =================================================================
    # STRATEGY 5: Global char substitution
    # =================================================================
    char_map: dict[str, str] = {}
    sub_ok = True
    for left, _, right, result in all_parsed:
        full = left + right
        if len(full) != len(result):
            sub_ok = False
            break
        for i in range(len(full)):
            if full[i] in char_map:
                if char_map[full[i]] != result[i]:
                    sub_ok = False
                    break
            else:
                char_map[full[i]] = result[i]
        if not sub_ok:
            break
    if sub_ok:
        t_full = t_left + t_right
        if all(c in char_map for c in t_full):
            return "".join(char_map[c] for c in t_full)

    raise ValueError("No programmatic pattern found for equation_transform")


# ===================================================================
# Registry
# ===================================================================

TOOL_REGISTRY: dict[str, callable] = {
    # General
    "eval_math": eval_math,
    "apply_formula": apply_formula,
    "round_number": round_number,
    "average": average,
    "regex_extract": regex_extract,
    # Bit manipulation
    "xor_binary": xor_binary,
    "and_binary": and_binary,
    "or_binary": or_binary,
    "not_binary": not_binary,
    "shift_left": shift_left,
    "shift_right": shift_right,
    "rotate_left": rotate_left,
    "rotate_right": rotate_right,
    # Cipher / substitution
    "substitute_chars": substitute_chars,
    "build_char_map": build_char_map,
    # Numeral conversion
    "to_roman": to_roman,
    "from_roman": from_roman,
    # Unit conversion / gravity
    "linear_factor": linear_factor,
    "compute_gravity_g": compute_gravity_g,
    "compute_gravity_d": compute_gravity_d,
    # End-to-end solvers
    "solve_gravity_physics": solve_gravity_physics,
    "solve_unit_conversion": solve_unit_conversion,
    "solve_numeral_conversion": solve_numeral_conversion,
    "solve_cipher_decryption": solve_cipher_decryption,
    "solve_bit_manipulation": solve_bit_manipulation,
    "solve_equation_transform": solve_equation_transform,
}


def run_tool(tool_name: str, tool_input: str) -> str:
    """Execute a named tool. Raises ValueError for unknown tools.
    Note: "ask_llm" is NOT in this registry -- it's handled by the solver.
    """
    func = TOOL_REGISTRY.get(tool_name)
    if func is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return func(tool_input)


# ===================================================================
# Tool descriptions (fed to the decompose LLM)
# ===================================================================

TOOL_DESCRIPTIONS = """\
Available tools (deterministic Python, no LLM, instant, 100% accurate):

GENERAL:
- ask_llm: Ask the LLM a question (for reasoning, pattern recognition, extraction).
  Input: {"question": "your question here"}
  Use for: identifying patterns, interpreting examples, any step needing language understanding.
  This is the ONLY tool that uses the LLM. All others below are pure computation.

- eval_math: Evaluate a math expression safely.
  Input: {"expr": "0.5 * 9.8 * 3**2"}

- apply_formula: Evaluate a formula with named variables.
  Input: {"formula": "0.5 * g * t ** 2", "vars": {"g": 9.8, "t": 3}}

- round_number: Round a number to n decimal places.
  Input: {"value": 154.6234, "decimals": 2}  ->  "154.62"

- average: Compute the mean of a list of numbers.
  Input: {"values": [15.88, 15.92, 15.87]}

- regex_extract: Extract all regex matches from text (returns JSON array).
  Input: {"text": "t = 1.37s, distance = 14.92 m", "pattern": "[\\\\d.]+"}

BIT MANIPULATION (8-bit binary):
- xor_binary: XOR two binary strings.  Input: {"a": "10110010", "b": "01001101"}
- and_binary: AND two binary strings.  Input: {"a": "10110010", "b": "01001101"}
- or_binary:  OR two binary strings.   Input: {"a": "10110010", "b": "01001101"}
- not_binary: Flip all bits.           Input: {"a": "10110010"}
- shift_left:  Left-shift by n.        Input: {"a": "10110010", "n": 1, "bits": 8}
- shift_right: Right-shift by n.       Input: {"a": "10110010", "n": 1, "bits": 8}
- rotate_left:  Circular rotate left.  Input: {"a": "10110010", "n": 1, "bits": 8}
- rotate_right: Circular rotate right. Input: {"a": "10110010", "n": 1, "bits": 8}

CIPHER / SUBSTITUTION:
- substitute_chars: Apply a character mapping to text.
  Input: {"text": "ucoov", "mapping": {"u": "q", "c": "u", "o": "e", "v": "n"}}

- build_char_map: Build substitution map from aligned (encrypted, plain) word pairs.
  Input: {"pairs": [["ucoov", "queen"], ["pqrsfv", "dragon"]]}
  Returns JSON mapping object.

NUMERAL CONVERSION (Roman numerals -- all puzzles in the dataset use Roman):
- to_roman: Convert integer to Roman numeral.     Input: {"number": 38}  ->  "XXXVIII"
- from_roman: Convert Roman numeral to integer.    Input: {"roman": "XXXVIII"}  ->  "38"

UNIT CONVERSION / GRAVITY PHYSICS:
- linear_factor: Compute average conversion factor from (input, output) pairs.
  Input: {"pairs": [[10.08, 6.69], [17.83, 11.83]]}

- compute_gravity_g: Compute g from (t, d) observations using g = 2d/t².
  Input: {"observations": [[1.37, 14.92], [4.27, 144.96]]}
  Returns the average g value.

- compute_gravity_d: Compute distance d = 0.5*g*t², rounded to 2 decimals.
  Input: {"g": 15.89, "t": 4.41}  ->  "154.62"\
"""
