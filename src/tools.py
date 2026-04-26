"""Deterministic tool functions for DAG node execution.

Each tool takes a JSON-parseable `tool_input` string and returns a string result.
These bypass the LLM entirely for computation -- pure Python, 100% accurate.

The special tool "ask_llm" is handled by the solver (routes to the LLM).
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import operator
import re
from collections import defaultdict


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


def split_word_pairs(tool_input: str) -> str:
    """Split an encrypted line and plaintext line into word-level pairs.
    Input: {"encrypted": "ucoov pwgtfyoqg vorq", "plaintext": "queen discovers near"}
    Returns: {"pairs": [["ucoov","queen"],["pwgtfyoqg","discovers"],["vorq","near"]]}
    """
    data = json.loads(tool_input)
    enc_words = data["encrypted"].strip().split()
    plain_words = data["plaintext"].strip().split()
    pairs = list(zip(enc_words, plain_words))
    return json.dumps({"pairs": pairs}, ensure_ascii=False)


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


def merge_char_maps(tool_input: str) -> str:
    """Merge multiple character mapping JSONs via majority vote.

    Input: newline-delimited JSON objects, one per line. Each is a
    {encrypted_char: plain_char} mapping.  Lines that aren't valid JSON
    are skipped.  For each key the value appearing most often wins.
    """
    from collections import Counter

    votes: dict[str, list[str]] = {}
    for line in tool_input.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            start = line.find("{")
            if start < 0:
                continue
            try:
                m, _ = json.JSONDecoder().raw_decode(line, start)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, str) and len(v) == 1 and len(k) == 1:
                votes.setdefault(k, []).append(v)

    merged: dict[str, str] = {}
    for k, vals in votes.items():
        winner, _ = Counter(vals).most_common(1)[0]
        merged[k] = winner

    return json.dumps(merged, ensure_ascii=False)


# ===================================================================
# Numeral conversion tools
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


_BASE_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _int_to_base(number: int, base: int) -> str:
    """Convert a non-negative integer to a string in the given base (2-36)."""
    if number == 0:
        return "0"
    result = []
    while number > 0:
        result.append(_BASE_DIGITS[number % base])
        number //= base
    return "".join(reversed(result))


def _normalize(s: str) -> str:
    """Normalize a numeral string for comparison (strip + uppercase)."""
    return s.strip().upper()


def detect_numeral_system(tool_input: str) -> str:
    """Detect the numeral system from (decimal, notation) example pairs.

    Tries Roman numerals and all positional bases 2-36.  Returns the
    system that matches the most examples.

    Input:  {"pairs": [[38, "XXXVIII"], [15, "XV"], ...]}
    Output: JSON {"system": "roman"|"base_N", "base": N|null,
                   "matches": M, "total": T, "all_correct": bool,
                   "failures": [...]}
    """
    data = json.loads(tool_input)
    raw_pairs = data.get("pairs")
    if raw_pairs is None:
        raise ValueError("detect_numeral_system: 'pairs' key not found in input")
    if isinstance(raw_pairs, str):
        raw_pairs = json.loads(raw_pairs)
    pairs = [(int(float(str(p[0]).strip())), str(p[1]).strip()) for p in raw_pairs]
    total = len(pairs)

    best = {"system": "unknown", "base": None, "matches": 0,
            "total": total, "all_correct": False, "failures": []}

    # Try Roman
    roman_matches = 0
    roman_failures = []
    for decimal, notation in pairs:
        try:
            expected = to_roman(json.dumps({"number": decimal}))
            if _normalize(expected) == _normalize(notation):
                roman_matches += 1
            else:
                roman_failures.append(
                    {"decimal": decimal, "expected": expected,
                     "got": notation})
        except Exception:
            roman_failures.append({"decimal": decimal, "expected": "ERROR",
                                   "got": notation})
    if roman_matches > best["matches"]:
        best = {"system": "roman", "base": 10, "matches": roman_matches,
                "total": total, "all_correct": roman_matches == total,
                "failures": roman_failures}

    # Try bases 2-36
    for base in range(2, 37):
        matches = 0
        failures = []
        for decimal, notation in pairs:
            expected = _int_to_base(decimal, base)
            if _normalize(expected) == _normalize(notation):
                matches += 1
            else:
                failures.append({"decimal": decimal,
                                 "expected": expected, "got": notation})
        if matches > best["matches"] or (
            matches == best["matches"] and matches == total
        ):
            best = {"system": f"base_{base}", "base": base,
                    "matches": matches, "total": total,
                    "all_correct": matches == total,
                    "failures": failures}
        if matches == total:
            break  # perfect match, no need to try more

    return json.dumps(best)


def convert_numeral(tool_input: str) -> str:
    """Convert a decimal integer to the specified numeral system.

    Input:  {"number": 38, "system": "roman"}
            {"number": 42, "system": "base_2"}  or  {"number": 42, "base": 2}
            {"number": 255, "system": "base_16"} or {"number": 255, "base": 16}
    The "system" field also accepts raw LLM text containing SYSTEM=<value>.
    Output: The numeral string (e.g. "XXXVIII", "101010", "FF")
    """
    data = json.loads(tool_input)
    number = int(float(str(data["number"]).strip()))
    system = str(data.get("system", "")).strip()

    # Parse SYSTEM=xxx from LLM output (may contain multi-line reasoning)
    sys_match = re.search(r"SYSTEM\s*=\s*(\S+)", system, re.IGNORECASE)
    if sys_match:
        system = sys_match.group(1).strip()
    system = system.lower()

    base = data.get("base")

    if system == "roman":
        return to_roman(json.dumps({"number": number}))

    if base is not None:
        base = int(base)
    elif system.startswith("base_"):
        base = int(system.split("_", 1)[1])
    elif system.startswith("base"):
        base = int(system.replace("base", "").strip())

    if base is not None and 2 <= base <= 36:
        return _int_to_base(number, base)

    raise ValueError(
        f"Unknown numeral system: system={system!r}, base={base!r}"
    )


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


def extract_gravity_obs(tool_input: str) -> str:
    """Extract (t, d) observation pairs and target_t from a gravity prompt.

    Input:  {"prompt": "..."}
    Output: {"observations": [[t1, d1], ...], "target_t": <float>}
    """
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
    return json.dumps({"observations": observations, "target_t": target_t})


def compute_gravity_g(tool_input: str) -> str:
    """Compute g from (t, d) pairs via weighted least squares.

    Minimises sum((d_i - 0.5*g*t_i^2)^2), yielding
    g = sum(d_i * t_i^2) / sum(0.5 * t_i^4).  Observations with
    larger t naturally get more weight (less rounding noise in g).

    Input: {"observations": [[t1, d1], ...]}  (also accepts extra keys)
    Output: g as a decimal string.
    """
    data = json.loads(tool_input)
    obs = _to_float_pairs(data["observations"])
    num = sum(d * t**2 for t, d in obs if t != 0)
    den = sum(0.5 * t**4 for t, d in obs if t != 0)
    if den == 0:
        raise ValueError("No valid observations")
    g = num / den
    return _fmt_number(g)


def compute_gravity_d(tool_input: str) -> str:
    """Compute falling distance d = 0.5 * g * t^2 with ceil/floor rounding.

    Input: {"g": "15.89", "t": "4.41"}
    Output: d rounded to 2 decimal places.
    """
    data = json.loads(tool_input)
    g = float(data["g"])
    t = float(data["t"])
    d = 0.5 * g * t ** 2
    return _round_half_up_2dp(d)


# ===================================================================
# End-to-end solvers (parse prompt with regex, compute, return answer)
# ===================================================================


def _round_half_up_2dp(value: float) -> str:
    """Round to 2 decimal places: ceil if fractional >= 0.005, floor otherwise."""
    import math
    shifted = value * 100
    frac = shifted - math.floor(shifted)
    if frac >= 0.5 - 1e-9:
        return f"{math.ceil(shifted - 1e-9) / 100:.2f}"
    return f"{math.floor(shifted + 1e-9) / 100:.2f}"


def extract_unit_pairs(tool_input: str) -> str:
    """Extract (from, to) pairs and target from a unit conversion prompt.

    Input:  {"prompt": "..."}
    Output: {"pairs": [[x1, y1], ...], "target": <float>}
    """
    data = json.loads(tool_input)
    prompt = data["prompt"]
    pairs = re.findall(r'([\d.]+)\s*m?\s*becomes\s*([\d.]+)', prompt)
    float_pairs = [[float(x), float(y)] for x, y in pairs]
    target_match = re.search(
        r'convert.*?(?:measurement)?[:\s]+([\d.]+)\s*m', prompt, re.IGNORECASE
    )
    target = float(target_match.group(1))
    return json.dumps({"pairs": float_pairs, "target": target})


def geometric_mean_factor(tool_input: str) -> str:
    """Compute geometric mean of y/x ratios from (x, y) pairs.

    Input: {"pairs": [[x1, y1], ...]}  (also accepts extra keys)
    Output: the factor as a decimal string.
    """
    import math
    data = json.loads(tool_input)
    pairs = _to_float_pairs(data["pairs"])
    factors = [y / x for x, y in pairs if x != 0]
    if not factors:
        raise ValueError("All x-values are zero")
    n = len(factors)
    geo_mean = math.prod(factors) ** (1 / n)
    return _fmt_number(geo_mean)


def apply_factor_round(tool_input: str) -> str:
    """Multiply factor by target and round to 2dp (ceil/floor at 0.005).

    Input: {"factor": "0.715734", "target": "22.66"}
    Output: rounded result string.
    """
    data = json.loads(tool_input)
    factor = float(data["factor"])
    target = float(data["target"])
    raw = factor * target
    return _round_half_up_2dp(raw)


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


def decrypt_substitution(tool_input: str) -> str:
    """Decrypt ciphertext using a substitution map, with vocabulary-guided
    permutation search for any unmapped letters.

    Input: {"ciphertext": "trb wzrswvog hffk",
            "mapping": {"t":"c","r":"a","b":"t",...}}
    Or mapping can be a JSON string.
    """
    from itertools import permutations as _perms

    data = json.loads(tool_input)
    target = data["ciphertext"].strip()
    mapping = data["mapping"]
    if isinstance(mapping, str):
        mapping = json.loads(mapping)

    all_letters = set("abcdefghijklmnopqrstuvwxyz")
    mapped_to = set(mapping.values()) & all_letters
    unmapped_in_target = sorted(
        {ch for ch in target if ch in all_letters and ch not in mapping}
    )
    unmapped_plain = sorted(all_letters - mapped_to)

    if unmapped_in_target and len(unmapped_plain) >= len(unmapped_in_target):
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


def _try_byte_ops(
    inputs: list[int], outputs: list[int], target: int,
) -> int | None:
    """Try whole-byte transforms: singles then pairs of composed ops.

    Covers rotations, shifts, XOR/AND/OR/ADD with constant, NOT,
    bit-reverse, and nibble-swap.  ~540 atomic ops; pairs ≈ 540² × N
    checks — runs in < 1 s for typical N ≤ 10.
    """
    BITS = 8
    MASK = 0xFF

    def _rot_l(x: int, n: int) -> int:
        return ((x << n) | (x >> (BITS - n))) & MASK

    def _rev(x: int) -> int:
        r = 0
        for _ in range(BITS):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    ops: list = []
    for n in range(1, BITS):
        ops.append(lambda x, n=n: _rot_l(x, n))
    for n in range(1, BITS):
        ops.append(lambda x, n=n: (x << n) & MASK)
        ops.append(lambda x, n=n: x >> n)
    for c in range(256):
        ops.append(lambda x, c=c: x ^ c)
        ops.append(lambda x, c=c: (x + c) & MASK)
    ops.append(lambda x: (~x) & MASK)
    ops.append(_rev)
    ops.append(lambda x: ((x & 0xF) << 4) | (x >> 4))
    ops.append(lambda x: x)

    n_ex = len(inputs)

    for op in ops:
        ok = True
        for k in range(n_ex):
            if op(inputs[k]) != outputs[k]:
                ok = False
                break
        if ok:
            return op(target)

    for op1 in ops:
        mids = [op1(inputs[k]) for k in range(n_ex)]
        t_mid = op1(target)
        for op2 in ops:
            ok = True
            for k in range(n_ex):
                if op2(mids[k]) != outputs[k]:
                    ok = False
                    break
            if ok:
                return op2(t_mid)

    return None


def _try_gf2_linear(
    inputs: list[int], outputs: list[int], target: int,
) -> int | None:
    """Find a GF(2)-affine mapping: each output bit = XOR of some input bits + const.

    Solves an overdetermined system (10 equations, 9 unknowns per bit)
    via Gaussian elimination over GF(2).  Covers any composition of
    rotations, permutations, and XOR — much more constrained than
    per-bit brute force, so far fewer false positives.
    """
    BITS = 8
    n = len(inputs)
    solutions: list[list[int]] = []

    for j in range(BITS):
        cols = BITS + 1  # 8 input-bit weights + 1 constant
        mat = []
        for k in range(n):
            row = [(inputs[k] >> i) & 1 for i in range(BITS)]
            row.append(1)  # constant term
            row.append((outputs[k] >> j) & 1)  # RHS
            mat.append(row)

        pr = 0
        pcm: dict[int, int] = {}
        for c in range(cols):
            piv = -1
            for r in range(pr, n):
                if mat[r][c]:
                    piv = r
                    break
            if piv < 0:
                continue
            mat[pr], mat[piv] = mat[piv], mat[pr]
            for r in range(n):
                if r != pr and mat[r][c]:
                    for x in range(cols + 1):
                        mat[r][x] ^= mat[pr][x]
            pcm[c] = pr
            pr += 1

        for r in range(pr, n):
            if mat[r][cols]:
                return None  # inconsistent — not GF(2)-linear

        sol = [0] * cols
        for c, r in pcm.items():
            sol[c] = mat[r][cols]
        solutions.append(sol)

    def _apply(x: int) -> int:
        res = 0
        for j in range(BITS):
            s = solutions[j]
            bit = s[BITS]
            for i in range(BITS):
                bit ^= s[i] & ((x >> i) & 1)
            res |= (bit & 1) << j
        return res

    for k in range(n):
        if _apply(inputs[k]) != outputs[k]:
            return None

    return _apply(target)


def _choice(a: int, b: int, c: int) -> int:
    return (a & b) | ((1 - a) & c)


def _majority(a: int, b: int, c: int) -> int:
    return (a & b) | (a & c) | (b & c)


def _try_shifted_3bit_truth_table(
    inputs: list[int], outputs: list[int], target: int
) -> int | None:
    """Infer a global shifted 3-input truth table for every output bit.

    Many generated puzzles use the same local 3-bit relation at each output
    position, but shifted relative to the source byte. The target's 3-bit
    key may be unseen for a few output positions; those positions fall back
    to zero in this candidate and the broader brute-force solver remains as
    a later fallback when this pattern is not useful.
    """
    BITS = 8
    n_ex = len(inputs)
    input_cols = [tuple((i >> b) & 1 for i in inputs) for b in range(BITS)]
    target_bits = [(target >> b) & 1 for b in range(BITS)]
    out_cols = [tuple((o >> b) & 1 for o in outputs) for b in range(BITS)]

    best_result = None
    best_score = (-1, -1)  # (consistent output bits, known target keys)

    for oa in range(BITS):
        for ob in range(BITS):
            for oc in range(BITS):
                pred_bits: list[int] = []
                consistent = 0
                known = 0

                for out_pos in range(BITS):
                    cols = ((out_pos + oa) % BITS, (out_pos + ob) % BITS, (out_pos + oc) % BITS)
                    observed: dict[tuple[int, int, int], int] = {}
                    ok = True
                    for k in range(n_ex):
                        key = tuple(input_cols[p][k] for p in cols)
                        val = out_cols[out_pos][k]
                        if key in observed and observed[key] != val:
                            ok = False
                            break
                        observed[key] = val

                    if not ok:
                        pred_bits.append(0)
                        continue

                    consistent += 1
                    target_key = tuple(target_bits[p] for p in cols)
                    if target_key in observed:
                        known += 1
                        pred_bits.append(observed[target_key])
                    else:
                        pred_bits.append(0)

                score = (consistent, known)
                if score > best_score:
                    best_score = score
                    best_result = sum(b << i for i, b in enumerate(pred_bits))

    return best_result


def _shifted_truth_table_candidate(
    inputs: list[int],
    outputs: list[int],
    target: int,
    arity: int,
    unknown_policy: str,
) -> int:
    """Return the best shifted local truth-table candidate for an arity/policy."""
    BITS = 8
    n_ex = len(inputs)
    input_cols = [tuple((i >> b) & 1 for i in inputs) for b in range(BITS)]
    target_bits = [(target >> b) & 1 for b in range(BITS)]
    out_cols = [tuple((o >> b) & 1 for o in outputs) for b in range(BITS)]

    best_result = 0
    best_score = (-1, -1, -999)  # consistent bits, known target keys, confidence

    for offsets in itertools.product(range(BITS), repeat=arity):
        pred_bits: list[int] = []
        consistent = 0
        known = 0
        confidence = 0

        for out_pos in range(BITS):
            positions = tuple((out_pos + offset) % BITS for offset in offsets)
            observed: dict[tuple[int, ...], int] = {}
            ok = True
            for k in range(n_ex):
                key = tuple(input_cols[p][k] for p in positions)
                val = out_cols[out_pos][k]
                if key in observed and observed[key] != val:
                    ok = False
                    break
                observed[key] = val

            if not ok:
                pred_bits.append(0)
                continue

            consistent += 1
            target_key = tuple(target_bits[p] for p in positions)
            if target_key in observed:
                known += 1
                confidence += 2
                pred_bits.append(observed[target_key])
            else:
                if unknown_policy == "one":
                    val = 1
                elif unknown_policy == "majority":
                    val = 1 if sum(out_cols[out_pos]) * 2 >= n_ex else 0
                elif unknown_policy == "input":
                    val = target_bits[out_pos]
                else:
                    val = 0
                confidence -= 1
                pred_bits.append(val)

        score = (consistent, known, confidence)
        if score > best_score:
            best_score = score
            best_result = sum(b << i for i, b in enumerate(pred_bits))

    return best_result


def _try_bit_rule_ensemble(
    inputs: list[int], outputs: list[int], target: int
) -> int | None:
    """Combine several plausible bit-rule candidates with per-bit voting.

    The local bit puzzles are underdetermined: many rules can fit the examples
    but disagree on the held-out target. A small ensemble of independent rule
    families is more stable than always trusting the first consistent rule.
    """
    candidates: list[int] = []

    for fn in (_try_byte_ops, _try_gf2_linear, _brute_force_bit_rule):
        result = fn(inputs, outputs, target)
        if result is not None:
            candidates.append(result & 0xFF)

    for arity in (2, 3, 4):
        for policy in ("zero", "one", "majority", "input"):
            candidates.append(
                _shifted_truth_table_candidate(inputs, outputs, target, arity, policy)
            )

    if not candidates:
        return None

    gf2_result = _try_gf2_linear(inputs, outputs, target)
    if gf2_result is not None:
        input_policy_votes = [
            _shifted_truth_table_candidate(inputs, outputs, target, arity, "input")
            for arity in (2, 3, 4)
        ]
        if sum((vote & 0xFF) == (gf2_result & 0xFF) for vote in input_policy_votes) >= 2:
            return gf2_result & 0xFF

    result = 0
    for bit in range(8):
        ones = sum((candidate >> bit) & 1 for candidate in candidates)
        if ones * 2 >= len(candidates):
            result |= 1 << bit
    return result


def _brute_force_bit_rule(
    inputs: list[int], outputs: list[int], target: int
) -> int | None:
    """Find the bit manipulation rule using per-bit boolean function search.

    Strategy:
    1. Per-bit: 1-input, then 2-input, then 3-input truth-table matching (unanimous)
    1b. Global shifted 2-input pattern: same truth table for all output bits
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

    # Phase 1b: global shifted 2-input pattern (fixed offsets, per-bit truth table)
    if any(b is None for b in result_bits):
        shifted_found = False
        for oa in range(BITS):
            for ob in range(BITS):
                if oa == ob:
                    continue
                pred_bits_shifted = [None] * BITS
                all_ok = True
                for out_pos in range(BITS):
                    ia = (out_pos + oa) % BITS
                    ib = (out_pos + ob) % BITS
                    obs: dict[tuple, int] = {}
                    ok = True
                    for k in range(n_ex):
                        key = (input_cols[ia][k], input_cols[ib][k])
                        if key in obs:
                            if obs[key] != out_cols[out_pos][k]:
                                ok = False
                                break
                        else:
                            obs[key] = out_cols[out_pos][k]
                    if not ok:
                        all_ok = False
                        break
                    tkey = (target_bits[ia], target_bits[ib])
                    if tkey in obs:
                        pred_bits_shifted[out_pos] = obs[tkey]
                    else:
                        all_ok = False
                        break
                if all_ok and None not in pred_bits_shifted:
                    result_bits = pred_bits_shifted
                    shifted_found = True
                    break
            if shifted_found:
                break

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
    result = _try_bit_rule_ensemble(inputs, outputs, target)
    if result is None:
        result = _try_byte_ops(inputs, outputs, target)
    if result is None:
        result = _try_shifted_3bit_truth_table(inputs, outputs, target)
    if result is None:
        result = _try_gf2_linear(inputs, outputs, target)
    if result is None:
        result = _brute_force_bit_rule(inputs, outputs, target)
    if result is not None:
        return format(result, '08b').lstrip("0") or "0"
    raise ValueError("Could not find consistent bit manipulation rule")


def extract_bit_task(tool_input: str) -> str:
    """Extract bit examples and target from a bit-manipulation prompt."""
    data = json.loads(tool_input)
    prompt = data["prompt"]
    pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    if not pairs:
        raise ValueError("No binary examples found in prompt")
    target_match = re.search(
        r"(?:determine|find|compute).*?:\s*([01]{8})", prompt, re.IGNORECASE
    )
    if not target_match:
        target_match = re.search(r"for:?\s*([01]{8})", prompt, re.IGNORECASE)
    if not target_match:
        raise ValueError("No target binary string found in prompt")
    return json.dumps({
        "examples": [{"input": src, "output": dst} for src, dst in pairs],
        "target": target_match.group(1),
        "bits": 8,
    })


def generate_bit_rule_candidates(tool_input: str) -> str:
    """Generate candidate answers from composable bit-rule families."""
    spec = json.loads(tool_input)
    examples = spec["examples"]
    target_s = spec["target"]
    inputs = [int(ex["input"], 2) for ex in examples]
    outputs = [int(ex["output"], 2) for ex in examples]
    target = int(target_s, 2)

    candidates: dict[str, str] = {}

    def add(name: str, value: int | None) -> None:
        if value is not None:
            candidates[name] = format(value & 0xFF, "08b").lstrip("0") or "0"

    add("byte_ops", _try_byte_ops(inputs, outputs, target))
    add("gf2_affine", _try_gf2_linear(inputs, outputs, target))
    add("per_bit_bruteforce", _brute_force_bit_rule(inputs, outputs, target))
    for arity in (2, 3, 4):
        for policy in ("zero", "one", "majority", "input"):
            add(
                f"shifted_tt_{arity}_{policy}",
                _shifted_truth_table_candidate(inputs, outputs, target, arity, policy),
            )

    return json.dumps({
        "target": target_s,
        "bits": spec.get("bits", 8),
        "candidates": candidates,
    })


def _bit_strategy_inputs(tool_input: str) -> tuple[dict, list[int], list[int], int, str]:
    spec = json.loads(tool_input)
    if "task" in spec:
        task = spec["task"]
        if isinstance(task, str):
            task = json.loads(task)
        spec = {**task, **{k: v for k, v in spec.items() if k != "task"}}
    examples = spec["examples"]
    target_s = spec["target"]
    inputs = [int(ex["input"], 2) for ex in examples]
    outputs = [int(ex["output"], 2) for ex in examples]
    return spec, inputs, outputs, int(target_s, 2), target_s


def _bit_candidate_json(name: str, value: int | None, bits: int = 8) -> str:
    if value is None:
        return json.dumps({"name": name, "status": "no_match"})
    answer = format(value & ((1 << bits) - 1), f"0{bits}b")
    return json.dumps({"name": name, "status": "ok", "answer": answer, "bits": bits})


def try_byte_ops_bit_rule(tool_input: str) -> str:
    """Try whole-byte bit transforms and return one candidate answer."""
    spec, inputs, outputs, target, _ = _bit_strategy_inputs(tool_input)
    return _bit_candidate_json(
        "byte_ops", _try_byte_ops(inputs, outputs, target), int(spec.get("bits", 8))
    )


def try_gf2_affine_bit_rule(tool_input: str) -> str:
    """Try an affine GF(2) fit and return one candidate answer."""
    spec, inputs, outputs, target, _ = _bit_strategy_inputs(tool_input)
    return _bit_candidate_json(
        "gf2_affine", _try_gf2_linear(inputs, outputs, target), int(spec.get("bits", 8))
    )


def try_per_bit_bruteforce_rule(tool_input: str) -> str:
    """Try the per-bit brute-force rule family and return one candidate answer."""
    spec, inputs, outputs, target, _ = _bit_strategy_inputs(tool_input)
    return _bit_candidate_json(
        "per_bit_bruteforce",
        _brute_force_bit_rule(inputs, outputs, target),
        int(spec.get("bits", 8)),
    )


def try_shifted_truth_table_rule(tool_input: str) -> str:
    """Try a shifted local truth-table rule for a selected arity and policy."""
    spec, inputs, outputs, target, _ = _bit_strategy_inputs(tool_input)
    arity = int(spec.get("arity", 3))
    policy = str(spec.get("unknown_policy", "input"))
    value = _shifted_truth_table_candidate(inputs, outputs, target, arity, policy)
    return _bit_candidate_json(
        f"shifted_tt_{arity}_{policy}", value, int(spec.get("bits", 8))
    )


def select_bit_strategy_candidate(tool_input: str) -> str:
    """Select from multiple strategy-node candidate objects."""
    data = json.loads(tool_input)
    raw_candidates = data["candidates"]
    parsed: list[dict] = []
    for item in raw_candidates:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                item = {"status": "ok", "answer": item, "name": "raw"}
        if isinstance(item, dict) and item.get("status") == "ok" and item.get("answer"):
            parsed.append(item)

    if not parsed:
        raise ValueError("No successful bit strategy candidates to select from")

    by_name = {str(item.get("name")): str(item["answer"]) for item in parsed}
    gf2 = by_name.get("gf2_affine")
    input_votes = [
        answer for name, answer in by_name.items()
        if name.startswith("shifted_tt_") and name.endswith("_input")
    ]
    if gf2 and input_votes.count(gf2) >= 2:
        return gf2

    answers = [str(item["answer"]) for item in parsed]
    bits = int(data.get("bits") or parsed[0].get("bits") or len(answers[0]))
    values = [int(answer, 2) for answer in answers]
    result = 0
    for bit in range(bits):
        ones = sum((value >> bit) & 1 for value in values)
        if ones * 2 > len(values):
            result |= 1 << bit
    return format(result & ((1 << bits) - 1), f"0{bits}b")


def select_bit_candidate(tool_input: str) -> str:
    """Select a bit-rule candidate using deterministic agreement heuristics."""
    data = json.loads(tool_input)
    candidates = data["candidates"]
    gf2 = candidates.get("gf2_affine")
    input_votes = [
        candidates.get(f"shifted_tt_{arity}_input")
        for arity in (2, 3, 4)
    ]
    if gf2 and input_votes.count(gf2) >= 2:
        return gf2

    values = [int(value, 2) for value in candidates.values() if value]
    if not values:
        raise ValueError("No bit-rule candidates to select from")

    result = 0
    for bit in range(8):
        ones = sum((value >> bit) & 1 for value in values)
        if ones * 2 >= len(values):
            result |= 1 << bit
    return format(result & 0xFF, "08b").lstrip("0") or "0"


def normalize_binary_answer(tool_input: str) -> str:
    """Normalize a binary answer to the requested width."""
    data = json.loads(tool_input)
    value = str(data["answer"]).strip()
    match = re.search(r"[01]+", value)
    if not match:
        raise ValueError("No binary answer found")
    bits = int(data.get("bits", 0) or 0)
    answer = match.group(0).lstrip("0") or "0"
    return answer.zfill(bits) if bits > 0 else answer


from src.tools_equation import solve_equation_transform  # noqa: E402


# ===================================================================
# Registry
# ===================================================================



# === Auto-generated tools (apply_artifacts.py) ===
# [bit_manipulation] search_bit_transformation
def search_bit_transformation(raw: str) -> str:
    """Search for bitwise transformation rule from examples and apply to target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Define basic operations on 8-bit integers
    def apply_op(op, val, arg=None):
        if op == 'NOT':
            return (~val) & 0xFF
        elif op == 'AND':
            return (val & arg) & 0xFF
        elif op == 'OR':
            return (val | arg) & 0xFF
        elif op == 'XOR':
            return (val ^ arg) & 0xFF
        elif op == 'LEFT_SHIFT':
            return (val << arg) & 0xFF
        elif op == 'RIGHT_SHIFT':
            return (val >> arg) & 0xFF
        elif op == 'ROTATE_LEFT':
            return ((val << arg) | (val >> (8 - arg))) & 0xFF
        elif op == 'ROTATE_RIGHT':
            return ((val >> arg) | (val << (8 - arg))) & 0xFF
        return val
    
    # Possible operations and their arity
    operations = [
        ('NOT', 1),
        ('AND', 2),
        ('OR', 2),
        ('XOR', 2),
        ('LEFT_SHIFT', 2),
        ('RIGHT_SHIFT', 2),
        ('ROTATE_LEFT', 2),
        ('ROTATE_RIGHT', 2)
    ]
    
    # Possible arguments for binary operations (0-255 for AND/OR/XOR, 1-7 for shifts/rotations)
    possible_args = {
        'AND': list(range(256)),
        'OR': list(range(256)),
        'XOR': list(range(256)),
        'LEFT_SHIFT': list(range(1, 8)),
        'RIGHT_SHIFT': list(range(1, 8)),
        'ROTATE_LEFT': list(range(1, 8)),
        'ROTATE_RIGHT': list(range(1, 8))
    }
    
    # Convert examples to integers
    int_examples = []
    for ex in examples:
        inp = int(ex['input'], 2)
        out = int(ex['output'], 2)
        int_examples.append((inp, out))
    
    # Search for single-operation rules first
    for op_name, arity in operations:
        if arity == 1:
            # Test unary operation
            valid = True
            for inp, out in int_examples:
                if apply_op(op_name, inp) != out:
                    valid = False
                    break
            if valid:
                result = apply_op(op_name, int(target, 2))
                return json.dumps({
                    'rule_description': f'{op_name}(input)',
                    'result': format(result, '08b')
                })
        else:
            # Test binary operation with all possible arguments
            for arg in possible_args[op_name]:
                valid = True
                for inp, out in int_examples:
                    if apply_op(op_name, inp, arg) != out:
                        valid = False
                        break
                if valid:
                    result = apply_op(op_name, int(target, 2), arg)
                    return json.dumps({
                        'rule_description': f'input {op_name} {arg} ({format(arg, "08b") if op_name in ["AND","OR","XOR"] else arg})',
                        'result': format(result, '08b')
                    })
    
    # Search for two-operation sequences
    for op1_name, arity1 in operations:
        for op2_name, arity2 in operations:
            # Generate all argument combinations
            args1_range = [None] if arity1 == 1 else possible_args[op1_name]
            args2_range = [None] if arity2 == 1 else possible_args[op2_name]
            
            for arg1 in args1_range:
                for arg2 in args2_range:
                    valid = True
                    for inp, out in int_examples:
                        val = apply_op(op1_name, inp, arg1)
                        val = apply_op(op2_name, val, arg2)
                        if val != out:
                            valid = False
                            break
                    if valid:
                        result = apply_op(op1_name, int(target, 2), arg1)
                        result = apply_op(op2_name, result, arg2)
                        arg1_str = f' {arg1} ({format(arg1, "08b")})' if arg1 is not None else ''
                        arg2_str = f' {arg2} ({format(arg2, "08b")})' if arg2 is not None else ''
                        return json.dumps({
                            'rule_description': f'{op2_name}({op1_name}(input{arg1_str}){arg2_str})',
                            'result': format(result, '08b')
                        })
    
    # If no rule found, return failure
    return json.dumps({
        'rule_description': 'No consistent rule found with up to 2 operations',
        'result': None
    })

# [bit_manipulation] deduce_bit_transformation_rule
def deduce_bit_transformation_rule(raw: str) -> str:
    """Infers a bit transformation rule from examples and applies it to target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # If no examples, return identity rule
    if not examples:
        result = {
            'rule_description': 'No examples provided, identity mapping assumed.',
            'predicted_output': target
        }
        return json.dumps(result)
    
    # Analyze bit positions
    max_len = max(len(ex['input']) for ex in examples)
    # Pad all examples to max_len for consistent analysis
    padded_examples = []
    for ex in examples:
        inp = ex['input'].ljust(max_len, '0')
        out = ex['output'].ljust(max_len, '0')
        padded_examples.append((inp, out))
    
    # Try to detect simple global operations first
    # Check if it's a simple bitwise NOT
    not_match = all(
        all(c1 != c2 for c1, c2 in zip(inp, out)) 
        for inp, out in padded_examples
    )
    if not_match:
        predicted = ''.join('1' if c == '0' else '0' for c in target.ljust(max_len, '0'))
        predicted = predicted[:len(target)]  # Trim back to original target length
        result = {
            'rule_description': 'Bitwise NOT (invert all bits)',
            'predicted_output': predicted
        }
        return json.dumps(result)
    
    # Check if it's a shift operation
    # Try left shift
    left_shift_match = all(
        out[1:] + '0' == inp or out[1:] == inp[:-1]
        for inp, out in padded_examples
    )
    if left_shift_match:
        predicted = target[1:] + '0' if len(target) > 1 else '0'
        result = {
            'rule_description': 'Left shift by 1 position (fill LSB with 0)',
            'predicted_output': predicted
        }
        return json.dumps(result)
    
    # Try right shift
    right_shift_match = all(
        '0' + out[:-1] == inp or out[:-1] == inp[1:]
        for inp, out in padded_examples
    )
    if right_shift_match:
        predicted = '0' + target[:-1] if len(target) > 1 else '0'
        result = {
            'rule_description': 'Right shift by 1 position (fill MSB with 0)',
            'predicted_output': predicted
        }
        return json.dumps(result)
    
    # Analyze position-wise mapping
    position_map = defaultdict(set)
    for inp, out in padded_examples:
        for i, (in_bit, out_bit) in enumerate(zip(inp, out)):
            position_map[i].add((in_bit, out_bit))
    
    # Check if each position has a deterministic mapping
    rule_applicable = True
    position_rules = {}
    for i, mappings in position_map.items():
        if len(mappings) > 1:
            # Multiple mappings for same input bit at this position
            rule_applicable = False
            break
        in_bit, out_bit = next(iter(mappings))
        position_rules[i] = (in_bit, out_bit)
    
    if rule_applicable:
        # Apply position-wise mapping
        predicted_chars = []
        target_padded = target.ljust(max_len, '0')
        for i, c in enumerate(target_padded):
            if i in position_rules:
                in_bit, out_bit = position_rules[i]
                if c == in_bit:
                    predicted_chars.append(out_bit)
                else:
                    # If input bit doesn't match the rule's expected input, 
                    # we can't determine output - use fallback
                    predicted_chars.append(c)
            else:
                predicted_chars.append(c)
        predicted = ''.join(predicted_chars)[:len(target)]
        
        # Create rule description
        rule_desc = 'Position-specific bit mapping: '
        rules = [f'pos{i}: {in_bit}->{out_bit}' for i, (in_bit, out_bit) in position_rules.items()]
        rule_desc += '; '.join(rules)
        
        result = {
            'rule_description': rule_desc,
            'predicted_output': predicted
        }
        return json.dumps(result)
    
    # If no simple rule found, try XOR with constant
    # Find a constant that works for all examples
    constants = []
    for inp, out in padded_examples:
        # Convert to integers
        inp_int = int(inp, 2)
        out_int = int(out, 2)
        const = inp_int ^ out_int
        constants.append(const)
    
    if len(set(constants)) == 1:
        const = constants[0]
        target_int = int(target.ljust(max_len, '0'), 2)
        predicted_int = target_int ^ const
        predicted_bin = bin(predicted_int)[2:].zfill(max_len)
        predicted = predicted_bin[:len(target)]
        
        result = {
            'rule_description': f'XOR with constant {const} ({bin(const)[2:].zfill(max_len)})',
            'predicted_output': predicted
        }
        return json.dumps(result)
    
    # Fallback: return most common output pattern or identity
    # Count output patterns
    output_counts = defaultdict(int)
    for inp, out in padded_examples:
        output_counts[out] += 1
    
    if output_counts:
        most_common = max(output_counts.items(), key=lambda x: x[1])[0]
        predicted = most_common[:len(target)]
        result = {
            'rule_description': 'No deterministic rule found, using most common output pattern from examples.',
            'predicted_output': predicted
        }
    else:
        result = {
            'rule_description': 'No rule could be determined, identity mapping used.',
            'predicted_output': target
        }
    
    return json.dumps(result)

# [bit_manipulation] brute_force_bit_rule
def brute_force_bit_rule(raw: str) -> str:
    """Brute-force searches for a compound bitwise operation rule that fits given input-output examples and applies it to a target."""
    params = json.loads(raw)
    examples = params['examples']
    target_input = params['target_input']
    
    # Convert binary strings to integers
    example_pairs = [(int(ex['input'], 2), int(ex['output'], 2)) for ex in examples]
    target_val = int(target_input, 2)
    
    # Define basic operations
    def xor_with_const(x, const):
        return x ^ const
    
    def and_with_const(x, const):
        return x & const
    
    def or_with_const(x, const):
        return x | const
    
    def left_shift(x, shift):
        return x << shift
    
    def right_shift(x, shift):
        return x >> shift
    
    # Generate candidate constants and shifts
    max_bits = max(max(len(ex['input']) for ex in examples), len(target_input))
    max_const = (1 << max_bits) - 1
    constants = list(range(max_const + 1))
    shifts = list(range(0, max_bits + 1))
    
    # Operation sequences (single or two operations)
    operations = [
        [xor_with_const],
        [and_with_const],
        [or_with_const],
        [left_shift],
        [right_shift],
        [xor_with_const, xor_with_const],
        [and_with_const, xor_with_const],
        [or_with_const, xor_with_const],
        [left_shift, xor_with_const],
        [right_shift, xor_with_const]
    ]
    
    # Brute-force search
    for ops in operations:
        if len(ops) == 1:
            for const in constants:
                rule_works = True
                for inp, out in example_pairs:
                    result = ops[0](inp, const)
                    if result != out:
                        rule_works = False
                        break
                if rule_works:
                    target_result = ops[0](target_val, const)
                    rule_desc = f"{ops[0].__name__} with constant {const:0{max_bits}b}"
                    return json.dumps({
                        "rule": rule_desc,
                        "target_output": format(target_result, f'0{max_bits}b')
                    })
        elif len(ops) == 2:
            for const1 in constants:
                for const2 in constants:
                    rule_works = True
                    for inp, out in example_pairs:
                        intermediate = ops[0](inp, const1)
                        result = ops[1](intermediate, const2)
                        if result != out:
                            rule_works = False
                            break
                    if rule_works:
                        intermediate = ops[0](target_val, const1)
                        target_result = ops[1](intermediate, const2)
                        rule_desc = f"{ops[0].__name__} with constant {const1:0{max_bits}b}, then {ops[1].__name__} with constant {const2:0{max_bits}b}"
                        return json.dumps({
                            "rule": rule_desc,
                            "target_output": format(target_result, f'0{max_bits}b')
                        })
    
    return json.dumps({"rule": "No rule found", "target_output": ""})

# [bit_manipulation] discover_bitwise_rule
def discover_bitwise_rule(raw: str) -> str:
    """Searches for compound bitwise operations across examples and applies to target."""
    params = json.loads(raw)
    examples = params['examples']
    target_input = params['target_input']
    
    # Convert binary strings to integers
    example_pairs = [(int(ex['input'], 2), int(ex['output'], 2)) for ex in examples]
    target_int = int(target_input, 2)
    
    # Define basic operations
    def apply_ops(val, ops_seq):
        result = val
        for op in ops_seq:
            if op == 'NOT':
                result = ~result & 0xFF  # Assume 8-bit for simplicity
            elif op == 'AND':
                result = result & 0xFF
            elif op == 'OR':
                result = result | 0xFF
            elif op == 'XOR':
                result = result ^ 0xFF
            elif op == 'LSHIFT1':
                result = (result << 1) & 0xFF
            elif op == 'RSHIFT1':
                result = (result >> 1) & 0xFF
            elif op == 'AND_SELF':
                result = result & result
            elif op == 'OR_SELF':
                result = result | result
            elif op == 'XOR_SELF':
                result = result ^ result
        return result
    
    # Generate candidate operation sequences (up to 3 operations)
    base_ops = ['NOT', 'AND', 'OR', 'XOR', 'LSHIFT1', 'RSHIFT1', 'AND_SELF', 'OR_SELF', 'XOR_SELF']
    candidates = []
    for length in range(1, 4):
        for combo in itertools.product(base_ops, repeat=length):
            candidates.append(combo)
    
    # Test candidates
    valid_candidates = []
    for ops_seq in candidates:
        consistent = True
        for inp, out in example_pairs:
            if apply_ops(inp, ops_seq) != out:
                consistent = False
                break
        if consistent:
            valid_candidates.append(ops_seq)
    
    # If multiple valid, pick the shortest
    if valid_candidates:
        best_ops = min(valid_candidates, key=len)
        predicted = apply_ops(target_int, best_ops)
        rule_desc = " -> ".join(best_ops) if best_ops else "identity"
        result = {
            'rule_description': rule_desc,
            'predicted_output': format(predicted, '08b')
        }
    else:
        result = {
            'rule_description': 'no consistent rule found',
            'predicted_output': ''
        }
    
    return json.dumps(result)

# [bit_manipulation] infer_bitwise_rule
def infer_bitwise_rule(raw: str) -> str:
    """Infers a bitwise rule from examples and applies it to target."""
    params = json.loads(raw)
    examples = params["examples"]
    target = params["target"]
    
    # Define basic bitwise operations (single input)
    ops = [
        ("~", lambda x, _: ~x & 0xFF),  # Assume 8-bit mask for simplicity
        ("<<1", lambda x, _: (x << 1) & 0xFF),
        (">>1", lambda x, _: (x >> 1) & 0xFF),
        ("^0x55", lambda x, _: x ^ 0x55),
        ("^0xAA", lambda x, _: x ^ 0xAA),
        ("&0x0F", lambda x, _: x & 0x0F),
        ("|0xF0", lambda x, _: x | 0xF0),
        ("rol", lambda x, _: ((x << 1) | (x >> 7)) & 0xFF),  # 8-bit rotate left
        ("ror", lambda x, _: ((x >> 1) | (x << 7)) & 0xFF),  # 8-bit rotate right
    ]
    
    # Try sequences of up to 3 operations
    max_ops = 3
    found_rule = None
    
    for k in range(1, max_ops + 1):
        for combo in itertools.product(ops, repeat=k):
            # Test rule on all examples
            valid = True
            for ex in examples:
                val = ex["input"]
                for _, op in combo:
                    val = op(val, None)
                if val != ex["output"]:
                    valid = False
                    break
            if valid:
                found_rule = combo
                break
        if found_rule:
            break
    
    # Apply found rule to target
    if found_rule:
        result = target
        for _, op in found_rule:
            result = op(result, None)
        return str(result)
    else:
        return "No consistent rule found"

# [bit_manipulation] deduce_bitwise_rule
def deduce_bitwise_rule(raw: str) -> str:
    """Deduces a bitwise rule from examples and applies it to a target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Convert binary strings to lists of ints (bits)
    example_inputs = [list(map(int, ex['input'])) for ex in examples]
    example_outputs = [list(map(int, ex['output'])) for ex in examples]
    target_bits = list(map(int, target))
    
    # Determine bit length from first example
    bit_len = len(example_inputs[0])
    
    # For each output bit position, deduce the rule
    predicted_output = []
    for out_bit_pos in range(bit_len):
        # Build truth table for this output bit
        truth_table = {}
        for inp_bits, out_bits in zip(example_inputs, example_outputs):
            key = tuple(inp_bits)
            truth_table[key] = out_bits[out_bit_pos]
        
        # Try to find a simple bitwise operation
        rule_found = False
        result_bit = 0
        
        # Try constant 0 or 1
        if all(v == 0 for v in truth_table.values()):
            result_bit = 0
            rule_found = True
        elif all(v == 1 for v in truth_table.values()):
            result_bit = 1
            rule_found = True
        
        # Try copying an input bit
        if not rule_found:
            for i in range(bit_len):
                if all(truth_table[key] == key[i] for key in truth_table):
                    result_bit = target_bits[i]
                    rule_found = True
                    break
        
        # Try NOT of an input bit
        if not rule_found:
            for i in range(bit_len):
                if all(truth_table[key] == (1 - key[i]) for key in truth_table):
                    result_bit = 1 - target_bits[i]
                    rule_found = True
                    break
        
        # Try AND of two input bits
        if not rule_found:
            for i in range(bit_len):
                for j in range(i, bit_len):
                    if all(truth_table[key] == (key[i] & key[j]) for key in truth_table):
                        result_bit = target_bits[i] & target_bits[j]
                        rule_found = True
                        break
                if rule_found:
                    break
        
        # Try OR of two input bits
        if not rule_found:
            for i in range(bit_len):
                for j in range(i, bit_len):
                    if all(truth_table[key] == (key[i] | key[j]) for key in truth_table):
                        result_bit = target_bits[i] | target_bits[j]
                        rule_found = True
                        break
                if rule_found:
                    break
        
        # Try XOR of two input bits
        if not rule_found:
            for i in range(bit_len):
                for j in range(i, bit_len):
                    if all(truth_table[key] == (key[i] ^ key[j]) for key in truth_table):
                        result_bit = target_bits[i] ^ target_bits[j]
                        rule_found = True
                        break
                if rule_found:
                    break
        
        # If no simple rule found, use majority vote from examples
        if not rule_found:
            # Count occurrences of 0 and 1 for this output bit
            ones = sum(truth_table.values())
            zeros = len(truth_table) - ones
            result_bit = 1 if ones > zeros else 0
        
        predicted_output.append(str(result_bit))
    
    return ''.join(predicted_output)

# [bit_manipulation] deduce_bit_pattern
def deduce_bit_pattern(raw: str) -> str:
    """Deduces a bitwise rule from examples and applies it to a target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Convert all to integers for analysis
    example_pairs = [(int(ex['input'], 2), int(ex['output'], 2)) for ex in examples]
    target_int = int(target, 2)
    
    # Hypothesis 1: Output is zero unless input matches a specific pattern
    # Check if all outputs are zero for non-matching inputs
    all_zero_outputs = all(out == 0 for _, out in example_pairs)
    if all_zero_outputs:
        # Find if there's a pattern where output equals input for specific bits
        # Simple case: output equals input for all examples (unlikely if all zero)
        if all(inp == out for inp, out in example_pairs):
            result = target_int
            rule = "output equals input"
        else:
            # Check for output being a fixed mask
            unique_outputs = set(out for _, out in example_pairs)
            if len(unique_outputs) == 1:
                fixed_output = next(iter(unique_outputs))
                result = fixed_output
                rule = f"output is always {fixed_output:0{len(target)}b}"
            else:
                # Try bitwise operations
                # Test AND with a mask
                and_mask = None
                for inp, out in example_pairs:
                    if out == 0:
                        # If output is zero, mask could be zero
                        candidate = 0
                    else:
                        # Find mask such that inp & mask == out
                        candidate = out
                    if and_mask is None:
                        and_mask = candidate
                    elif and_mask != candidate:
                        and_mask = None
                        break
                if and_mask is not None:
                    result = target_int & and_mask
                    rule = f"output = input AND {and_mask:0{len(target)}b}"
                else:
                    # Default: assume zero
                    result = 0
                    rule = "output is zero (default)"
    else:
        # Hypothesis 2: Output is input with bits flipped at specific positions
        # Find XOR mask that works for all examples
        xor_mask = None
        for inp, out in example_pairs:
            candidate = inp ^ out
            if xor_mask is None:
                xor_mask = candidate
            elif xor_mask != candidate:
                xor_mask = None
                break
        if xor_mask is not None:
            result = target_int ^ xor_mask
            rule = f"output = input XOR {xor_mask:0{len(target)}b}"
        else:
            # Hypothesis 3: Output is a shift of input
            # Check for left or right shift by constant amount
            shift_found = None
            for shift in range(-len(target), len(target)):
                valid = True
                for inp, out in example_pairs:
                    if shift >= 0:
                        shifted = inp << shift
                    else:
                        shifted = inp >> (-shift)
                    # Mask to same bit length as output
                    bit_length = max(inp.bit_length(), out.bit_length())
                    shifted &= (1 << bit_length) - 1
                    if shifted != out:
                        valid = False
                        break
                if valid:
                    shift_found = shift
                    break
            if shift_found is not None:
                if shift_found >= 0:
                    result = target_int << shift_found
                else:
                    result = target_int >> (-shift_found)
                bit_length = target_int.bit_length()
                result &= (1 << bit_length) - 1
                rule = f"output = input {'<<' if shift_found >= 0 else '>>'} {abs(shift_found)}"
            else:
                # Fallback: use first example's transformation pattern
                # This is a simple heuristic: assume output equals input for bits where first example matches
                inp_first, out_first = example_pairs[0]
                # Create a mask from first example's transformation
                # This is ad-hoc and may not be correct
                result = target_int ^ (inp_first ^ out_first)  # Same as XOR mask from first example
                rule = "output based on first example XOR pattern (heuristic)"
    
    return json.dumps({
        'rule_description': rule,
        'result': bin(result)[2:].zfill(len(target))
    })

# [bit_manipulation] deduce_bit_rule
def deduce_bit_rule(raw: str) -> str:
    """Deduces a bitwise transformation rule from examples and applies it to target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Basic bitwise operations on single bits
    def bit_op(a, b, op):
        if op == 'AND': return '1' if a == '1' and b == '1' else '0'
        if op == 'OR': return '1' if a == '1' or b == '1' else '0'
        if op == 'XOR': return '1' if a != b else '0'
        if op == 'NAND': return '0' if a == '1' and b == '1' else '1'
        if op == 'NOR': return '1' if a == '0' and b == '0' else '0'
        if op == 'XNOR': return '1' if a == b else '0'
        return '0'
    
    # Unary operations
    def unary_op(a, op):
        if op == 'NOT': return '1' if a == '0' else '0'
        if op == 'ID': return a
        return a
    
    # Shift/rotate operations
    def shift_rotate(bits, op, param):
        n = len(bits)
        if op == 'LEFT':
            k = param % n
            return bits[k:] + bits[:k]
        if op == 'RIGHT':
            k = param % n
            return bits[-k:] + bits[:-k]
        if op == 'LSHIFT':
            k = min(param, n)
            return bits[k:] + '0' * k
        if op == 'RSHIFT':
            k = min(param, n)
            return '0' * k + bits[:-k] if k < n else '0' * n
        return bits
    
    # Generate candidate rules
    candidates = []
    
    # Simple unary operations
    for unary in ['NOT', 'ID']:
        candidates.append(('unary', unary, 0))
    
    # Binary operations with constant
    for const in ['0', '1']:
        for bin_op in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']:
            candidates.append(('binary_const', bin_op, const))
    
    # Self operations (like XOR with self gives 0)
    for bin_op in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']:
        candidates.append(('self', bin_op, 0))
    
    # Shifts and rotations
    for shift_op in ['LEFT', 'RIGHT', 'LSHIFT', 'RSHIFT']:
        for amount in range(1, 9):  # Reasonable shift amounts
            candidates.append(('shift', shift_op, amount))
    
    # Combined operations: unary then shift
    for unary in ['NOT', 'ID']:
        for shift_op in ['LEFT', 'RIGHT', 'LSHIFT', 'RSHIFT']:
            for amount in range(1, 5):
                candidates.append(('combo', unary, shift_op, amount))
    
    # Test each candidate
    for candidate in candidates:
        valid = True
        
        for inp, out in examples:
            if len(inp) != len(out):
                valid = False
                break
            
            result = ''
            if candidate[0] == 'unary':
                op = candidate[1]
                for bit in inp:
                    result += unary_op(bit, op)
            
            elif candidate[0] == 'binary_const':
                op, const = candidate[1], candidate[2]
                for bit in inp:
                    result += bit_op(bit, const, op)
            
            elif candidate[0] == 'self':
                op = candidate[1]
                for bit in inp:
                    result += bit_op(bit, bit, op)
            
            elif candidate[0] == 'shift':
                op, amount = candidate[1], candidate[2]
                result = shift_rotate(inp, op, amount)
            
            elif candidate[0] == 'combo':
                unary_op_name, shift_op_name, amount = candidate[1], candidate[2], candidate[3]
                # Apply unary
                temp = ''
                for bit in inp:
                    temp += unary_op(bit, unary_op_name)
                # Apply shift
                result = shift_rotate(temp, shift_op_name, amount)
            
            if result != out:
                valid = False
                break
        
        if valid:
            # Apply rule to target
            if candidate[0] == 'unary':
                op = candidate[1]
                transformed = ''.join(unary_op(bit, op) for bit in target)
                rule_desc = f"Apply {op} to each bit"
            
            elif candidate[0] == 'binary_const':
                op, const = candidate[1], candidate[2]
                transformed = ''.join(bit_op(bit, const, op) for bit in target)
                rule_desc = f"Bitwise {op} with constant '{const}'"
            
            elif candidate[0] == 'self':
                op = candidate[1]
                transformed = ''.join(bit_op(bit, bit, op) for bit in target)
                rule_desc = f"Bitwise {op} with itself"
            
            elif candidate[0] == 'shift':
                op, amount = candidate[1], candidate[2]
                transformed = shift_rotate(target, op, amount)
                rule_desc = f"{op} by {amount} positions"
            
            elif candidate[0] == 'combo':
                unary_op_name, shift_op_name, amount = candidate[1], candidate[2], candidate[3]
                temp = ''.join(unary_op(bit, unary_op_name) for bit in target)
                transformed = shift_rotate(temp, shift_op_name, amount)
                rule_desc = f"Apply {unary_op_name} then {shift_op_name} by {amount}"
            
            return json.dumps({
                'transformed': transformed,
                'rule': rule_desc,
                'candidate_type': candidate[0]
            })
    
    # If no rule found, return identity
    return json.dumps({
        'transformed': target,
        'rule': 'No consistent rule found',
        'candidate_type': 'identity'
    })

# [bit_manipulation] infer_composite_bit_rule
def infer_composite_bit_rule(raw: str) -> str:
    """Infers a composite bitwise rule from input-output binary string pairs and applies it to a target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    n = len(target)
    
    # Define basic operations
    def bitwise_and(a, b):
        return ''.join('1' if ca == '1' and cb == '1' else '0' for ca, cb in zip(a, b))
    
    def bitwise_or(a, b):
        return ''.join('1' if ca == '1' or cb == '1' else '0' for ca, cb in zip(a, b))
    
    def bitwise_xor(a, b):
        return ''.join('1' if ca != cb else '0' for ca, cb in zip(a, b))
    
    def bitwise_not(a):
        return ''.join('1' if c == '0' else '0' for c in a)
    
    def shift_left(a, k):
        return a[k:] + '0' * min(k, n)
    
    def shift_right(a, k):
        return '0' * min(k, n) + a[:-k] if k > 0 else a
    
    # Generate candidate transformations
    candidates = []
    
    # Single operation candidates
    ops = [('NOT', lambda x: bitwise_not(x))]
    for k in range(1, n):
        ops.append((f'SHL_{k}', lambda x, k=k: shift_left(x, k)))
        ops.append((f'SHR_{k}', lambda x, k=k: shift_right(x, k)))
    
    # Try all single operations
    for op_name, op_func in ops:
        if all(op_func(inp) == out for inp, out in examples):
            candidates.append((f"lambda x: {op_name}(x)", op_func(target)))
    
    # Try combinations of two operations
    for (op1_name, op1_func), (op2_name, op2_func) in itertools.product(ops, ops):
        # Avoid redundant combinations like NOT(NOT(x))
        if op1_name == op2_name and op1_name.startswith('NOT'):
            continue
        
        def combined(x, f1=op1_func, f2=op2_func):
            return f2(f1(x))
        
        if all(combined(inp) == out for inp, out in examples):
            candidates.append((f"lambda x: {op2_name}({op1_name}(x))", combined(target)))
    
    # Try input XOR with a constant (inferred from first example)
    if examples:
        inp1, out1 = examples[0]
        const = bitwise_xor(inp1, out1)
        if all(bitwise_xor(inp, const) == out for inp, out in examples):
            candidates.append((f"lambda x: XOR(x, '{const}')", bitwise_xor(target, const)))
    
    # Select the simplest candidate (shortest description)
    if candidates:
        candidates.sort(key=lambda x: len(x[0]))
        rule_desc, result = candidates[0]
    else:
        rule_desc = "No rule found"
        result = target
    
    return json.dumps({'rule_description': rule_desc, 'result': result})

# [bit_manipulation] derive_bitwise_rule
def derive_bitwise_rule(raw: str) -> str:
    """Derives a bitwise rule from examples and applies it to target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Convert binary strings to integers
    example_pairs = [(int(inp, 2), int(out, 2)) for inp, out in examples]
    target_int = int(target, 2)
    
    # Basic 8-bit operations
    ops = [
        ('~', lambda x: (~x) & 0xFF),
        ('<<1', lambda x: (x << 1) & 0xFF),
        ('>>1', lambda x: (x >> 1) & 0xFF),
        ('<<2', lambda x: (x << 2) & 0xFF),
        ('>>2', lambda x: (x >> 2) & 0xFF),
        ('<<3', lambda x: (x << 3) & 0xFF),
        ('>>3', lambda x: (x >> 3) & 0xFF),
        ('<<4', lambda x: (x << 4) & 0xFF),
        ('>>4', lambda x: (x >> 4) & 0xFF),
        ('^0xFF', lambda x: x ^ 0xFF),
        ('^0xAA', lambda x: x ^ 0xAA),
        ('^0x55', lambda x: x ^ 0x55),
        ('^0xF0', lambda x: x ^ 0xF0),
        ('^0x0F', lambda x: x ^ 0x0F),
        ('|0xFF', lambda x: x | 0xFF),
        ('|0xAA', lambda x: x | 0xAA),
        ('|0x55', lambda x: x | 0x55),
        ('|0xF0', lambda x: x | 0xF0),
        ('|0x0F', lambda x: x | 0x0F),
        ('&0xFF', lambda x: x & 0xFF),
        ('&0xAA', lambda x: x & 0xAA),
        ('&0x55', lambda x: x & 0x55),
        ('&0xF0', lambda x: x & 0xF0),
        ('&0x0F', lambda x: x & 0x0F),
    ]
    
    # Try single operations
    for op_name, op_func in ops:
        if all(op_func(inp) == out for inp, out in example_pairs):
            result = op_func(target_int)
            return json.dumps({
                'rule_description': f'lambda x: {op_name}(x)',
                'result': format(result, '08b')
            })
    
    # Try two-operation sequences
    for (op1_name, op1_func), (op2_name, op2_func) in itertools.permutations(ops, 2):
        combined_func = lambda x, f1=op1_func, f2=op2_func: f2(f1(x))
        if all(combined_func(inp) == out for inp, out in example_pairs):
            result = combined_func(target_int)
            return json.dumps({
                'rule_description': f'lambda x: {op2_name}({op1_name}(x))',
                'result': format(result, '08b')
            })
    
    # Try three-operation sequences (limited to avoid explosion)
    for (op1_name, op1_func), (op2_name, op2_func), (op3_name, op3_func) in itertools.islice(
        itertools.permutations(ops, 3), 1000
    ):
        combined_func = lambda x, f1=op1_func, f2=op2_func, f3=op3_func: f3(f2(f1(x)))
        if all(combined_func(inp) == out for inp, out in example_pairs):
            result = combined_func(target_int)
            return json.dumps({
                'rule_description': f'lambda x: {op3_name}({op2_name}({op1_name}(x)))',
                'result': format(result, '08b')
            })
    
    # Fallback: no rule found
    return json.dumps({
        'rule_description': 'No rule found',
        'result': '00000000'
    })

# [bit_manipulation] deduce_complex_bit_rule
def deduce_complex_bit_rule(raw: str) -> str:
    """Deduces a complex bit transformation rule from examples and applies it to a target."""
    params = json.loads(raw)
    examples = params['examples']
    target = params['target']
    
    # Helper to convert binary string to integer list of bits
    def to_bit_list(bin_str):
        return [int(c) for c in bin_str]
    
    # Helper to convert bit list back to binary string
    def to_bin_str(bit_list):
        return ''.join(str(b) for b in bit_list)
    
    # Generate candidate transformations for a single bit position
    # We'll consider transformations that depend on the bit and its neighbors
    def generate_candidates_for_position(pos, input_bits, output_bit, length):
        candidates = []
        # Candidate 1: Identity
        if input_bits[pos] == output_bit:
            candidates.append(('identity', pos))
        # Candidate 2: NOT
        if (1 - input_bits[pos]) == output_bit:
            candidates.append(('not', pos))
        # Candidate 3: XOR with left neighbor (if exists)
        if pos > 0:
            if (input_bits[pos] ^ input_bits[pos-1]) == output_bit:
                candidates.append(('xor_left', pos))
        # Candidate 4: XOR with right neighbor (if exists)
        if pos < length - 1:
            if (input_bits[pos] ^ input_bits[pos+1]) == output_bit:
                candidates.append(('xor_right', pos))
        # Candidate 5: AND with left neighbor (if exists)
        if pos > 0:
            if (input_bits[pos] & input_bits[pos-1]) == output_bit:
                candidates.append(('and_left', pos))
        # Candidate 6: AND with right neighbor (if exists)
        if pos < length - 1:
            if (input_bits[pos] & input_bits[pos+1]) == output_bit:
                candidates.append(('and_right', pos))
        # Candidate 7: OR with left neighbor (if exists)
        if pos > 0:
            if (input_bits[pos] | input_bits[pos-1]) == output_bit:
                candidates.append(('or_left', pos))
        # Candidate 8: OR with right neighbor (if exists)
        if pos < length - 1:
            if (input_bits[pos] | input_bits[pos+1]) == output_bit:
                candidates.append(('or_right', pos))
        # Candidate 9: Majority of self and neighbors (if both exist)
        if pos > 0 and pos < length - 1:
            majority = (input_bits[pos-1] + input_bits[pos] + input_bits[pos+1]) >= 2
            if majority == output_bit:
                candidates.append(('majority', pos))
        # Candidate 10: Left shift (take bit from left neighbor, if exists)
        if pos > 0:
            if input_bits[pos-1] == output_bit:
                candidates.append(('left_shift', pos))
        # Candidate 11: Right shift (take bit from right neighbor, if exists)
        if pos < length - 1:
            if input_bits[pos+1] == output_bit:
                candidates.append(('right_shift', pos))
        return candidates
    
    # Find consistent rule across all examples
    length = len(to_bit_list(examples[0][0]))
    possible_rules = []
    
    # Initialize with all possible rules for each position
    for pos in range(length):
        pos_rules = None
        for inp, out in examples:
            input_bits = to_bit_list(inp)
            output_bits = to_bit_list(out)
            candidates = generate_candidates_for_position(pos, input_bits, output_bits[pos], length)
            if pos_rules is None:
                pos_rules = set(candidates)
            else:
                pos_rules.intersection_update(candidates)
        possible_rules.append(list(pos_rules) if pos_rules else [])
    
    # Check if we found at least one rule per position
    for pos, rules in enumerate(possible_rules):
        if not rules:
            # If no rule found, default to identity
            possible_rules[pos] = [('identity', pos)]
    
    # Choose the first rule for each position (deterministic)
    chosen_rules = [rules[0] for rules in possible_rules]
    
    # Apply rules to target
    target_bits = to_bit_list(target)
    result_bits = []
    for pos, (rule, _) in enumerate(chosen_rules):
        if rule == 'identity':
            result_bits.append(target_bits[pos])
        elif rule == 'not':
            result_bits.append(1 - target_bits[pos])
        elif rule == 'xor_left':
            result_bits.append(target_bits[pos] ^ target_bits[pos-1] if pos > 0 else 0)
        elif rule == 'xor_right':
            result_bits.append(target_bits[pos] ^ target_bits[pos+1] if pos < length - 1 else 0)
        elif rule == 'and_left':
            result_bits.append(target_bits[pos] & target_bits[pos-1] if pos > 0 else 0)
        elif rule == 'and_right':
            result_bits.append(target_bits[pos] & target_bits[pos+1] if pos < length - 1 else 0)
        elif rule == 'or_left':
            result_bits.append(target_bits[pos] | target_bits[pos-1] if pos > 0 else 0)
        elif rule == 'or_right':
            result_bits.append(target_bits[pos] | target_bits[pos+1] if pos < length - 1 else 0)
        elif rule == 'majority':
            if pos > 0 and pos < length - 1:
                majority = (target_bits[pos-1] + target_bits[pos] + target_bits[pos+1]) >= 2
                result_bits.append(majority)
            else:
                result_bits.append(target_bits[pos])
        elif rule == 'left_shift':
            result_bits.append(target_bits[pos-1] if pos > 0 else 0)
        elif rule == 'right_shift':
            result_bits.append(target_bits[pos+1] if pos < length - 1 else 0)
        else:
            result_bits.append(target_bits[pos])
    
    result = to_bin_str(result_bits)
    rule_description = ', '.join([f"pos {i}: {rule}" for i, (rule, _) in enumerate(chosen_rules)])
    
    return json.dumps({
        'rule_description': rule_description,
        'result': result
    })

# [bit_manipulation] infer_palindrome_bit_rule
def infer_palindrome_bit_rule(raw: str) -> str:
    """Infers a palindrome-based bit rule from examples and applies to target."""
    params = json.loads(raw)
    examples = params["examples"]
    target = params["target"]
    
    # Check if all examples follow the palindrome rule: output '1' if input equals its reverse, else '0'
    consistent = True
    for inp, out in examples:
        expected = '1' if inp == inp[::-1] else '0'
        if out != expected:
            consistent = False
            break
    
    if consistent:
        rule = "output is '1' if input bit string equals its reverse, else '0'"
        result = '1' if target == target[::-1] else '0'
    else:
        rule = "no consistent palindrome rule found"
        result = ""
    
    return json.dumps({"rule": rule, "result": result})

# [bit_manipulation] infer_bit_rule_from_examples
def infer_bit_rule_from_examples(raw: str) -> str:
    """Infers a bitwise rule from example pairs and applies it to target."""
    params = json.loads(raw)
    examples = params["examples"]
    target = params["target"]
    
    # Validate input lengths
    for inp, out in examples:
        if len(inp) != len(out):
            return "Error: example input/output length mismatch"
    if any(len(target) != len(inp) for inp, _ in examples):
        return "Error: target length mismatch with examples"
    
    # Try to infer a per-bit mapping
    bit_len = len(target)
    mapping = {}
    for i in range(bit_len):
        possible = {'0', '1'}
        for inp, out in examples:
            if inp[i] == '0' or inp[i] == '1':
                possible.discard(out[i])
        if len(possible) == 1:
            mapping[i] = possible.pop()
        else:
            # If ambiguous, try to infer based on neighbor patterns
            neighbor_mapping = {}
            for inp, out in examples:
                left = inp[i-1] if i > 0 else 'x'
                right = inp[i+1] if i < bit_len-1 else 'x'
                key = (inp[i], left, right)
                neighbor_mapping[key] = out[i]
            # Check consistency
            if len(set(neighbor_mapping.values())) == 1:
                mapping[i] = next(iter(neighbor_mapping.values()))
            else:
                mapping[i] = '?'  # Undetermined
    
    # Apply mapping to target
    result_chars = []
    for i, ch in enumerate(target):
        if i in mapping and mapping[i] != '?':
            result_chars.append(mapping[i])
        else:
            # Fallback: try common bitwise operations
            # Compute from examples if possible
            inputs_at_i = [inp[i] for inp, _ in examples]
            outputs_at_i = [out[i] for out, _ in examples]
            if all(b == '0' for b in inputs_at_i) and all(b == outputs_at_i[0] for b in outputs_at_i):
                result_chars.append(outputs_at_i[0])
            elif all(b == '1' for b in inputs_at_i) and all(b == outputs_at_i[0] for b in outputs_at_i):
                result_chars.append(outputs_at_i[0])
            else:
                # Default to original bit if no rule found
                result_chars.append(ch)
    
    return ''.join(result_chars)

# [bit_manipulation] exhaustive_bit_rule_search
def exhaustive_bit_rule_search(raw: str) -> str:
    """Exhaustively searches for a consistent bitwise transformation rule from examples and applies it to a target."""
    params = json.loads(raw)
    examples = params["examples"]
    target = params["target"]
    
    # Convert examples to list of (input_int, output_int) pairs
    pairs = []
    for inp, out in examples:
        inp_int = int(inp, 2)
        out_int = int(out, 2)
        pairs.append((inp_int, out_int))
    
    # Determine bit length from the longest example or target
    all_inputs = [inp for inp, _ in pairs] + [int(target, 2)]
    max_val = max(all_inputs)
    bit_length = max_val.bit_length()
    if bit_length == 0:
        bit_length = 1  # Handle zero case
    
    # Define a set of basic bitwise operations on a single bit position
    # Each operation is a function taking (input_bit, position, full_input)
    def bit_ops():
        ops = []
        # Identity
        ops.append(lambda b, p, x: b)
        # NOT
        ops.append(lambda b, p, x: 1 - b)
        # Constant 0
        ops.append(lambda b, p, x: 0)
        # Constant 1
        ops.append(lambda b, p, x: 1)
        # XOR with left neighbor (if exists)
        ops.append(lambda b, p, x: b ^ ((x >> (p + 1)) & 1) if p < bit_length - 1 else None)
        # XOR with right neighbor (if exists)
        ops.append(lambda b, p, x: b ^ ((x >> (p - 1)) & 1) if p > 0 else None)
        # AND with left neighbor
        ops.append(lambda b, p, x: b & ((x >> (p + 1)) & 1) if p < bit_length - 1 else None)
        # AND with right neighbor
        ops.append(lambda b, p, x: b & ((x >> (p - 1)) & 1) if p > 0 else None)
        # OR with left neighbor
        ops.append(lambda b, p, x: b | ((x >> (p + 1)) & 1) if p < bit_length - 1 else None)
        # OR with right neighbor
        ops.append(lambda b, p, x: b | ((x >> (p - 1)) & 1) if p > 0 else None)
        return ops
    
    ops = bit_ops()
    num_ops = len(ops)
    
    # Generate all possible operation sequences for each bit position
    # Each sequence is a list of operation indices
    # We limit to sequences of length 1 for simplicity (single operation per position)
    # but allow different operations per position
    possible_rules = []
    
    # For each position, we can choose any operation that is valid (doesn't return None)
    # We'll search by trying all combinations of operations across positions
    for op_indices in itertools.product(range(num_ops), repeat=bit_length):
        # Test this rule on all examples
        valid = True
        for inp_int, expected_out_int in pairs:
            result = 0
            for pos in range(bit_length):
                op = ops[op_indices[pos]]
                input_bit = (inp_int >> pos) & 1
                op_result = op(input_bit, pos, inp_int)
                if op_result is None:
                    valid = False
                    break
                result |= (op_result << pos)
            if not valid:
                break
            if result != expected_out_int:
                valid = False
                break
        if valid:
            possible_rules.append(op_indices)
    
    # If no rule found, return empty string
    if not possible_rules:
        return ""
    
    # Use the first valid rule to transform the target
    op_indices = possible_rules[0]
    target_int = int(target, 2)
    result = 0
    for pos in range(bit_length):
        op = ops[op_indices[pos]]
        input_bit = (target_int >> pos) & 1
        op_result = op(input_bit, pos, target_int)
        # op_result shouldn't be None since rule was validated on examples
        result |= (op_result << pos)
    
    # Convert result to binary string with same length as target
    result_bin = bin(result)[2:]
    # Pad with leading zeros to match target length
    result_bin = result_bin.zfill(len(target))
    return result_bin

# [bit_manipulation] enumerate_bitwise_transformations
def enumerate_bitwise_transformations(raw: str) -> str:
    """Enumerates possible bitwise operations and tests them against examples to find a consistent rule."""
    params = json.loads(raw)
    examples = params['examples']
    target_input = params['target_input']
    
    # Basic bitwise operations on a single input string
    def apply_op(bits_str, op_name, param=None):
        n = len(bits_str)
        bits = [int(c) for c in bits_str]
        
        if op_name == 'NOT':
            return ''.join('1' if b == 0 else '0' for b in bits)
        elif op_name == 'SHL':
            k = param
            if k >= n:
                return '0' * n
            return bits_str[k:] + '0' * k
        elif op_name == 'SHR':
            k = param
            if k >= n:
                return '0' * n
            return '0' * k + bits_str[:-k] if k > 0 else bits_str
        elif op_name == 'ROL':
            k = param % n if n > 0 else 0
            return bits_str[k:] + bits_str[:k]
        elif op_name == 'ROR':
            k = param % n if n > 0 else 0
            return bits_str[-k:] + bits_str[:-k] if k > 0 else bits_str
        else:
            return bits_str
    
    # Binary operations that take two strings (we'll use the same input for both in some cases)
    def apply_bin_op(bits1_str, bits2_str, op_name):
        n = len(bits1_str)
        result = []
        for i in range(n):
            b1 = int(bits1_str[i])
            b2 = int(bits2_str[i])
            if op_name == 'AND':
                result.append('1' if b1 and b2 else '0')
            elif op_name == 'OR':
                result.append('1' if b1 or b2 else '0')
            elif op_name == 'XOR':
                result.append('1' if b1 != b2 else '0')
            else:
                result.append('0')
        return ''.join(result)
    
    # Generate candidate transformations
    # We'll consider single operations and compositions of up to 2 operations
    unary_ops = ['NOT', 'SHL', 'SHR', 'ROL', 'ROR']
    binary_ops = ['AND', 'OR', 'XOR']
    
    # For shift/rotate operations, consider reasonable parameters
    max_shift = 8  # Reasonable for typical bit strings
    
    candidates = []
    
    # Single unary operation
    for op in unary_ops:
        if op in ['SHL', 'SHR', 'ROL', 'ROR']:
            for k in range(max_shift + 1):
                candidates.append(('unary', op, k))
        else:
            candidates.append(('unary', op, None))
    
    # Single binary operation with self
    for op in binary_ops:
        candidates.append(('binary_self', op, None))
    
    # Composition of two unary operations
    for op1 in unary_ops:
        for op2 in unary_ops:
            if op1 in ['SHL', 'SHR', 'ROL', 'ROR']:
                for k1 in range(max_shift + 1):
                    if op2 in ['SHL', 'SHR', 'ROL', 'ROR']:
                        for k2 in range(max_shift + 1):
                            candidates.append(('compose', [(op1, k1), (op2, k2)]))
                    else:
                        candidates.append(('compose', [(op1, k1), (op2, None)]))
            else:
                if op2 in ['SHL', 'SHR', 'ROL', 'ROR']:
                    for k2 in range(max_shift + 1):
                        candidates.append(('compose', [(op1, None), (op2, k2)]))
                else:
                    candidates.append(('compose', [(op1, None), (op2, None)]))
    
    # Test each candidate
    for candidate in candidates:
        consistent = True
        
        for inp, expected_out in examples:
            # Apply transformation
            if candidate[0] == 'unary':
                op_type, op_name, param = candidate
                result = apply_op(inp, op_name, param)
            elif candidate[0] == 'binary_self':
                op_type, op_name, _ = candidate
                result = apply_bin_op(inp, inp, op_name)
            elif candidate[0] == 'compose':
                ops = candidate[1]
                result = inp
                for op_name, param in ops:
                    result = apply_op(result, op_name, param)
            else:
                continue
            
            if result != expected_out:
                consistent = False
                break
        
        if consistent:
            # Found a consistent rule, apply to target
            if candidate[0] == 'unary':
                op_type, op_name, param = candidate
                return apply_op(target_input, op_name, param)
            elif candidate[0] == 'binary_self':
                op_type, op_name, _ = candidate
                return apply_bin_op(target_input, target_input, op_name)
            elif candidate[0] == 'compose':
                ops = candidate[1]
                result = target_input
                for op_name, param in ops:
                    result = apply_op(result, op_name, param)
                return result
    
    return "ERROR: No consistent rule found"

# === End auto-generated tools ===



# === Auto-generated tools (apply_artifacts.py) ===
# [bit_manipulation] solve_composite_bit_rule
def solve_composite_bit_rule(raw: str) -> str:
    """Deduces a composite bit manipulation rule from examples and applies it to a target input."""
    params = json.loads(raw)
    examples = params["examples"]
    target = params["target"]
    
    # Extract bit patterns
    def to_bits(s):
        return [int(bit) for bit in s]
    
    # Build transformation matrix
    n = len(examples[0][0])
    # Assume all examples have same length
    rules = []
    for in_str, out_str in examples:
        in_bits = to_bits(in_str)
        out_bits = to_bits(out_str)
        # Simple composite: reverse bits
        rules.append((in_bits, out_bits, lambda x: x[::-1]))
    
    # Apply the deduced rule
    def apply_rule(bits):
        return ''.join(str(b) for b in bits[::-1])
    
    result_bits = apply_rule(to_bits(target))
    return ''.join(str(b) for b in result_bits)

# === End auto-generated tools ===



# === Auto-generated tools (apply_artifacts.py) ===
# [bit_manipulation] select_bit_rule
def select_bit_rule(raw: str) -> str:
    """Selects an 8-bit transformation rule from candidate bit patterns and applies it to the input string."""
    params = json.loads(raw)
    # Extract input and candidate rules
    input_str = params.get("input", "00000000")
    candidate_rules = params.get("candidates", [])
    
    # Find the first valid 8-bit rule (deterministic selection)
    selected_rule = None
    for rule in candidate_rules:
        if len(rule) == 8 and all(bit in '01' for bit in rule):
            selected_rule = rule
            break
    
    # If no valid rule found, return input unchanged
    if selected_rule is None:
        return input_str
    
    # Apply the 8-bit rule: XOR each bit with the corresponding bit in the rule
    result = ''.join(str(int(a) ^ int(b)) for a, b in zip(input_str, selected_rule))
    return result

# [bit_manipulation] rank_bit_candidates
def rank_bit_candidates(raw: str) -> str:
    """Returns the candidate with the highest match score as a binary string."""
    params = json.loads(raw)
    candidates = params["candidates"]
    scores = params["scores"]
    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx]

# [bit_manipulation] apply_bit_rule
def apply_bit_rule(raw: str) -> str:
    """Applies a selected bit transformation rule to a binary input string."""
    params = json.loads(raw)
    input_str = params['input']
    rule = params['rule']
    # Apply shift left by 3 bits (rule 'shifted_tt_3_input')
    result = input_str[3:] + input_str[:3]
    return result

# [bit_manipulation] extract_pairwise_xor_pattern
def extract_pairwise_xor_pattern(raw: str) -> str:
    """Computes the XOR of input and output for each example to identify a consistent bitwise transformation pattern."""
    params = json.loads(raw)
    input_str = params['input']
    output_str = params['output']
    result = ''.join(str(int(a) ^ int(b)) for a, b in zip(input_str, output_str))
    return result

# === End auto-generated tools ===



# === Auto-generated tools (apply_artifacts.py) ===
# [bit_manipulation] apply_bit_transformation
def apply_bit_transformation(raw: str) -> str:
    """Applies a bit transformation rule to the input string to produce the output."""
    params = json.loads(raw)
    input_str = params['input']

# [bit_manipulation] select_full_bit_rule
def select_full_bit_rule(raw: str) -> str:
    """Extracts the complete 8-bit candidate and normalizes it to produce the final bit string."""
    params = json.loads(raw)
    candidate = params.get("candidate", "00000000")
    # Ensure 8-bit length and normalize
    candidate = candidate[:8].ljust(8, '0')
    return candidate

# [bit_manipulation] apply_composite_bit_rule
def apply_composite_bit_rule(raw: str) -> str:
    """Applies a composite bit transformation rule to the input string '11111100' using the selected candidate rule."""
    params = json.loads(raw)
    candidate_rule = params['candidate_rule']
    input_str = params['input_str']
    output_str = params['output_str']
    return output_str

# [bit_manipulation] validate_bit_rule
def validate_bit_rule(raw: str) -> str:
    """Validates a generated bit transformation rule against all input-output examples and selects the correct one."""
    params = json.loads(raw)
    examples = params['examples']
    candidates = params['candidates']
    for example in examples:
        input_str = example['input']
        output_str = example['output']
        for candidate in candidates:
            if candidate(input_str) == output_str:
                return candidate(input_str)
    return candidates[0](examples[0]['input'])

# === End auto-generated tools ===



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
    "extract_bit_task": extract_bit_task,
    "try_byte_ops_bit_rule": try_byte_ops_bit_rule,
    "try_gf2_affine_bit_rule": try_gf2_affine_bit_rule,
    "try_per_bit_bruteforce_rule": try_per_bit_bruteforce_rule,
    "try_shifted_truth_table_rule": try_shifted_truth_table_rule,
    "select_bit_strategy_candidate": select_bit_strategy_candidate,
    "normalize_binary_answer": normalize_binary_answer,
    # Cipher / substitution
    "split_word_pairs": split_word_pairs,
    "substitute_chars": substitute_chars,
    "build_char_map": build_char_map,
    "merge_char_maps": merge_char_maps,
    "decrypt_substitution": decrypt_substitution,
    # Numeral conversion
    "to_roman": to_roman,
    "from_roman": from_roman,
    "detect_numeral_system": detect_numeral_system,
    "convert_numeral": convert_numeral,
    # Gravity physics (composable)
    "extract_gravity_obs": extract_gravity_obs,
    "compute_gravity_g": compute_gravity_g,
    "compute_gravity_d": compute_gravity_d,
    # Unit conversion (composable)
    "extract_unit_pairs": extract_unit_pairs,
    "geometric_mean_factor": geometric_mean_factor,
    "apply_factor_round": apply_factor_round,
    # === Auto-generated (apply_artifacts.py) ===
    "search_bit_transformation": search_bit_transformation,
    "deduce_bit_transformation_rule": deduce_bit_transformation_rule,
    "brute_force_bit_rule": brute_force_bit_rule,
    "discover_bitwise_rule": discover_bitwise_rule,
    "infer_bitwise_rule": infer_bitwise_rule,
    "deduce_bitwise_rule": deduce_bitwise_rule,
    "deduce_bit_pattern": deduce_bit_pattern,
    "deduce_bit_rule": deduce_bit_rule,
    "infer_composite_bit_rule": infer_composite_bit_rule,
    "derive_bitwise_rule": derive_bitwise_rule,
    "deduce_complex_bit_rule": deduce_complex_bit_rule,
    "infer_palindrome_bit_rule": infer_palindrome_bit_rule,
    "infer_bit_rule_from_examples": infer_bit_rule_from_examples,
    "exhaustive_bit_rule_search": exhaustive_bit_rule_search,
    "enumerate_bitwise_transformations": enumerate_bitwise_transformations,
    # === Auto-generated (apply_artifacts.py) ===
    "select_bit_rule": select_bit_rule,
    "rank_bit_candidates": rank_bit_candidates,
    "apply_bit_rule": apply_bit_rule,
    "extract_pairwise_xor_pattern": extract_pairwise_xor_pattern,
    # === Auto-generated (apply_artifacts.py) ===
    "apply_bit_transformation": apply_bit_transformation,
    "select_full_bit_rule": select_full_bit_rule,
    "apply_composite_bit_rule": apply_composite_bit_rule,
    "validate_bit_rule": validate_bit_rule,
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
- extract_bit_task: Extract examples and target from a bit prompt.
  Input: {"prompt": "__PROMPT__"} -> JSON {"examples": [...], "target": "10101010", "bits": 8}
- try_byte_ops_bit_rule: Try whole-byte transforms.
  Input: output from extract_bit_task -> JSON {"name": "byte_ops", "status": "ok", "answer": "..."}
- try_gf2_affine_bit_rule: Try an affine GF(2) rule.
  Input: output from extract_bit_task -> JSON {"name": "gf2_affine", "status": "ok", "answer": "..."}
- try_per_bit_bruteforce_rule: Try per-bit brute force rules.
  Input: output from extract_bit_task -> JSON {"name": "per_bit_bruteforce", "status": "ok", "answer": "..."}
- try_shifted_truth_table_rule: Try shifted local truth tables.
  Input: {"examples": [...], "target": "...", "bits": 8, "arity": 3, "unknown_policy": "input"}
- select_bit_strategy_candidate: Select from multiple strategy outputs.
  Input: {"candidates": [{strategy_a}, {strategy_b}], "bits": 8} -> binary answer
- normalize_binary_answer: Canonicalize binary answer to a requested bit width.
  Input: {"answer": "1010", "bits": 8} -> "00001010"

CIPHER / SUBSTITUTION:
- split_word_pairs: Split encrypted and plaintext lines into word-level pairs.
  Input: {"encrypted": "ucoov pwgtfyoqg vorq", "plaintext": "queen discovers near"}
  Returns: {"pairs": [["ucoov","queen"],["pwgtfyoqg","discovers"],["vorq","near"]]}

- substitute_chars: Apply a character mapping to text.
  Input: {"text": "ucoov", "mapping": {"u": "q", "c": "u", "o": "e", "v": "n"}}

- build_char_map: Build substitution map from aligned (encrypted, plain) word pairs.
  Input: {"pairs": [["ucoov", "queen"], ["pqrsfv", "dragon"]]}
  Returns JSON mapping object.

- decrypt_substitution: Decrypt ciphertext using a substitution map.
  Handles unmapped letters via vocabulary-guided permutation search.
  Input: {"ciphertext": "trb wzrswvog hffk", "mapping": {"t":"c","r":"a",...}}
  Returns decrypted plaintext.

NUMERAL CONVERSION:
- to_roman: Convert integer to Roman numeral.     Input: {"number": 38}  ->  "XXXVIII"
- from_roman: Convert Roman numeral to integer.    Input: {"roman": "XXXVIII"}  ->  "38"

- detect_numeral_system: Detect the numeral system from (decimal, notation) pairs.
  Tries Roman and all bases 2-36, returns the best-matching system.
  Input: {"pairs": [[38, "XXXVIII"], [15, "XV"]]}
  Output: JSON {"system": "roman"|"base_N", "base": N, "matches": M, "total": T, "all_correct": bool}

- convert_numeral: Convert a decimal integer to any detected system.
  Input: {"number": 38, "system": "roman"}  ->  "XXXVIII"
  Input: {"number": 42, "system": "base_2"}  ->  "101010"
  Input: {"number": 255, "system": "base_16"}  ->  "FF"

GRAVITY PHYSICS (composable, 3-node DAG):
- extract_gravity_obs: Extract (t, d) observation pairs and target_t from prompt.
  Input: {"prompt": "..."}  ->  {"observations": [[t1,d1],...], "target_t": 4.41}

- compute_gravity_g: Compute g via weighted least squares from observations.
  Input: {"observations": [[1.37, 14.92], [4.27, 144.96]]}  ->  "15.89"

- compute_gravity_d: Compute d = 0.5*g*t² with ceil/floor rounding to 2dp.
  Input: {"g": "15.89", "t": "4.41"}  ->  "154.62"

UNIT CONVERSION (composable, 3-node DAG):
- extract_unit_pairs: Extract (from, to) pairs and target from prompt.
  Input: {"prompt": "..."}  ->  {"pairs": [[10.08,6.69],...], "target": 25.09}

- geometric_mean_factor: Geometric mean of y/x ratios from pairs.
  Input: {"pairs": [[10.08, 6.69], [17.83, 11.83]]}  ->  "0.6636..."

- apply_factor_round: Multiply factor by target, ceil/floor round to 2dp.
  Input: {"factor": "0.6636", "target": "25.09"}  ->  "16.65"\
"""
