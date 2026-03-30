from src.state import GraphState

PUZZLE_SIGNATURES: dict[str, str] = {
    "bit manipulation": "bit_manipulation",
    "numeral system": "numeral_conversion",
    "unit conversion": "unit_conversion",
    "encryption rules": "cipher_decryption",
    "transformation rules": "equation_transform",
    "gravitational constant": "gravity_physics",
}


def classify_node(state: GraphState) -> dict:
    prompt_lower = state["prompt"].lower()
    for signature, puzzle_type in PUZZLE_SIGNATURES.items():
        if signature in prompt_lower:
            return {"puzzle_type": puzzle_type}
    return {"puzzle_type": "unknown"}
