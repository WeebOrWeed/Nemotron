"""Equation-transform solver.

Extracted from tools.py so a collaborator can work on it independently.
The single public entry point ``solve_equation_transform`` is registered
in ``TOOL_REGISTRY`` via an import in ``tools.py``.
"""

from __future__ import annotations

import json
import math


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
    # STRATEGY 4b: Symbolic per-char ordinal (variable-length results)
    # =================================================================
    if same_op and len(same_op) >= 2:
        for r_len in sorted(set(len(r) for _, _, r in same_op)):
            subset = [(l, r, res) for l, r, res in same_op if len(res) == r_len]
            if len(subset) < 2:
                continue
            for M in [94, 95, 127, 128, 256]:
                for off in [0, 33, 32, -33, -32]:
                    src_pairs = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
                    for combo in src_pairs[:]:
                        if r_len >= 2:
                            src_pairs_2 = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
                        else:
                            src_pairs_2 = src_pairs
                    per_char_ops_list = [
                        lambda a, b: a + b,
                        lambda a, b: a - b,
                        lambda a, b: b - a,
                        lambda a, b: a ^ b,
                        lambda a, b: a | b,
                        lambda a, b: a & b,
                    ]

                    def get_operand_chars(l, r):
                        return [ord(l[0]), ord(l[1]), ord(r[0]), ord(r[1])]

                    best_pos_maps = []
                    for j in range(r_len):
                        found_j = False
                        for si in range(4):
                            for sj in range(4):
                                for op_fn in per_char_ops_list:
                                    ok = True
                                    for l, r, res in subset:
                                        sc = get_operand_chars(l, r)
                                        try:
                                            v = (op_fn(sc[si], sc[sj]) + off) % M
                                            if v < 0 or v > 127 or chr(v) != res[j]:
                                                ok = False
                                                break
                                        except (ValueError, ZeroDivisionError):
                                            ok = False
                                            break
                                    if ok:
                                        best_pos_maps.append((si, sj, op_fn))
                                        found_j = True
                                        break
                                if found_j:
                                    break
                            if found_j:
                                break
                        if not found_j:
                            break
                    if len(best_pos_maps) == r_len:
                        tc = get_operand_chars(t_left, t_right)
                        try:
                            result_chars = []
                            for si, sj, op_fn in best_pos_maps:
                                v = (op_fn(tc[si], tc[sj]) + off) % M
                                result_chars.append(chr(v))
                            return "".join(result_chars)
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
