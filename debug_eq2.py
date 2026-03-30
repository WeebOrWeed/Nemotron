"""Test: is equation_transform a cryptarithmetic puzzle?"""
import itertools

# Puzzle 00457d26:
# `!*[{ = '"[`     (op: *, pos 2)
# \'*'> = ![@      (op: *, pos 2)
# \'-!` = \\       (op: -, pos 2)
# `!*\& = '@'{     (op: *, pos 2)
# Target: [[-!' → @&  (op: -, pos 2)

# Characters: ` ! [ { ' " > @ \ &
# 10 unique chars → digits 0-9

# Parse equations: left_op right = result
equations = [
    ("`!", "*", "[{", "'\"[`"),    # 2-digit * 2-digit = 4-digit
    ("\\'", "*", "'>", "![@"),     # 2-digit * 2-digit = 3-digit
    ("\\'", "-", "!`", "\\\\"),    # 2-digit - 2-digit = 2-digit
    ("`!", "*", "\\&", "'@'{"),    # 2-digit * 2-digit = 4-digit
]
target = ("[[", "-", "!'")

chars = list(set("".join([e[0]+e[2]+e[3] for e in equations]) + target[0] + target[2]))
chars.sort()
print(f"Unique chars: {chars} (count: {len(chars)})")

def to_num(s, mapping):
    n = 0
    for c in s:
        n = n * 10 + mapping[c]
    return n

def check(mapping):
    for left, op, right, result in equations:
        l = to_num(left, mapping)
        r = to_num(right, mapping)
        res = to_num(result, mapping)
        if op == "*":
            if l * r != res:
                return False
        elif op == "-":
            if l - r != res:
                return False
        elif op == "+":
            if l + r != res:
                return False
    return True

# Brute force
# Optimization: equation 3 says \' - !` = \\ → 11*D[\]
# So (10*D[\]+D[']) - (10*D[!]+D[`]) = 11*D[\]
# → D['] - 10*D[!] - D[`] = D[\]
# → D['] = D[\] + 10*D[!] + D[`]
# Since D['] is a single digit (0-9), 10*D[!] must be 0, so D[!]=0
# Then D['] = D[\] + D[`]

print("From equation 3: D[!] must be 0, and D['] = D[\\] + D[`]")
print()

# With D[!]=0, try remaining assignments
remaining_chars = [c for c in chars if c != '!']
remaining_digits = [d for d in range(10) if d != 0]

solutions = []
count = 0

for perm in itertools.permutations(remaining_digits, len(remaining_chars)):
    mapping = {'!': 0}
    for i, c in enumerate(remaining_chars):
        mapping[c] = perm[i]
    # Quick check: D['] = D[\] + D[`]
    if mapping["'"] != mapping["\\"] + mapping["`"]:
        continue
    count += 1
    if check(mapping):
        solutions.append(dict(mapping))
        t_left = to_num(target[0], mapping)
        t_right = to_num(target[2], mapping)
        if target[1] == "-":
            result = t_left - t_right
        elif target[1] == "*":
            result = t_left * t_right
        elif target[1] == "+":
            result = t_left + t_right
        # Convert back to chars
        inv = {v: k for k, v in mapping.items()}
        result_str = ""
        for d in str(result):
            result_str += inv[int(d)]
        print(f"Solution found! Mapping: {mapping}")
        print(f"  {target[0]} {target[1]} {target[2]} = {result} = '{result_str}'")

print(f"\nChecked {count} pruned permutations")
print(f"Found {len(solutions)} solutions")
