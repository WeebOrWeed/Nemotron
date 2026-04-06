# Nemotron DAG-of-Thoughts Pipeline

Spec-oriented design for the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
Uses a DAG-of-Thoughts architecture to decompose complex reasoning puzzles
into smaller sub-questions, execute them with dependency-aware parallelism,
and retry on failure.

## Architecture

```
START
  │
  ▼
classify   (keyword match + build deterministic DAG via QueryPlanner — zero LLM calls)
  │          outputs: puzzle_type, thought_dag (sub-questions + tool assignments)
  ▼
decompose  (first pass: forwards classifier's DAG unchanged)
  │          (on retry: LLM generates a new DAG from failure context)
  ▼
solve_next (find ready nodes, run on threads, collect answers)
  │         │                          │
  │ unsolved nodes remain              │ node failed, retries < MAX
  │ (no failure)                       │
  ▼                                    ▼
solve_next ◄──────────────────── decompose (re-decompose with failure context)
  │
  │ DAG complete OR retries exhausted
  ▼
 END
```

### Conditional edges from `solve_next`

| Condition                             | Target      |
|---------------------------------------|-------------|
| Unsolved nodes remain, no failure     | `solve_next` (loop) |
| Node failed, `retries < MAX_RETRIES`  | `decompose` (retry) |
| DAG complete or retries exhausted     | `END`       |

## The 6 Puzzle Types

Each type is identified by a deterministic keyword match on the prompt
(no LLM inference, zero latency). The classifier also emits a full
execution plan (sub-questions + tool assignments) so the downstream
decompose node can use it directly.  Distribution is roughly even (~1,550
each in the training set).

| Type                 | Signature phrase                              | Task                                          |
|----------------------|-----------------------------------------------|-----------------------------------------------|
| `bit_manipulation`   | "bit manipulation"                            | solve_bit_manipulation: brute-force per-bit boolean function search (fully deterministic) |
| `cipher_decryption`  | "encryption rules"                            | N parallel [split_word_pairs → build_char_map] → merge_char_maps → decrypt_substitution (fully deterministic) |
| `equation_transform` | "transformation rules"                        | solve_equation_transform: operator-centric compound-operation search (fully deterministic) |
| `gravity_physics`    | "gravitational constant"                      | solve_gravity_physics: regex-extract (t,d) pairs, per-pair g, geometric mean, d=0.5*g*t² (fully deterministic) |
| `numeral_conversion` | "numeral system"                              | solve_numeral_conversion: detect Roman numerals and convert target (fully deterministic) |
| `unit_conversion`    | "unit conversion"                             | solve_unit_conversion: regex-extract pairs, majority-vote factor, multiply (fully deterministic) |

## DAG-of-Thoughts

Each puzzle is decomposed into a **Directed Acyclic Graph** of thought
nodes. Each node is a simple sub-question with explicit parent dependencies.

### ThoughtNode

```python
class ThoughtNode(TypedDict):
    id: str                  # e.g. "try_xor"
    question: str            # the sub-question (may contain {parent_id} placeholders)
    depends_on: list[str]    # IDs of parent nodes whose answers are needed
    tool: Optional[str]      # if set, run this deterministic tool instead of LLM
    tool_input: Optional[str]  # JSON string with tool parameters ({parent_id} interpolated)
    answer: Optional[str]    # filled by solver
```

- `depends_on: []` = root node, can execute immediately.
- `depends_on: ["a", "b"]` = blocked until both "a" and "b" have answers.
- `tool: None` = LLM answers the question. `tool: "xor_binary"` = run the Python function.
- Parent answers are interpolated into both `question` and `tool_input` via `{parent_id}`.
- JSON sub-field access is supported: if a parent's answer is JSON, `{parent_id_field}`
  expands to `answer["field"]` (e.g. `{linear_fit_slope}` from `{"slope":0.92,...}`).

### Execution Strategy: Fully Deterministic

All six puzzle types are solved with **zero LLM calls** during DAG
execution. Each type has a deterministic solver in `tools.py` that
handles the full puzzle end-to-end. The `QueryPlanner` builds
single-node (or multi-node for cipher) DAGs that invoke these solvers
directly.

| Type                 | DAG Structure                                                       |
|----------------------|---------------------------------------------------------------------|
| `gravity_physics`    | Single node: `solve_gravity_physics` — regex-extract (t,d) pairs, per-pair g, geometric mean, d=0.5×g×t² |
| `unit_conversion`    | Single node: `solve_unit_conversion` — regex-extract pairs & target, majority-vote factor, multiply |
| `numeral_conversion` | Single node: `solve_numeral_conversion` — detect Roman numerals, convert target |
| `cipher_decryption`  | N parallel [split_word_pairs → build_char_map] → merge_char_maps → decrypt_substitution |
| `bit_manipulation`   | Single node: `solve_bit_manipulation` — brute-force per-bit boolean function search |
| `equation_transform` | Single node: `solve_equation_transform` — operator-centric compound-operation search |

If a deterministic solver fails, `solver` still uses an LLM retry fallback.

### QueryPlanner (all 6 types — fully deterministic)

All six puzzle types use the `QueryPlanner` (`src/planner.py`) to build
DAGs at classify time. No LLM calls are made during planning — each
type has a `_build_*_dag` function that constructs ThoughtNodes directly:

- **`gravity_physics`**: single `solve_gravity_physics` node
- **`unit_conversion`**: single `solve_unit_conversion` node
- **`numeral_conversion`**: single `solve_numeral_conversion` node
- **`bit_manipulation`**: single `solve_bit_manipulation` node
- **`equation_transform`**: single `solve_equation_transform` node
- **`cipher_decryption`**: multi-node pipeline (N parallel extract+map → merge → translate)

#### Cipher decryption — per-case parallel mapping

For `cipher_decryption`, the entire pipeline is deterministic — no LLM
calls.  Routes are extracted via regex, then each example is processed
in parallel.

```
extract_pairs_1 (split_word_pairs) → create_mapping_1 (build_char_map) ─┐
extract_pairs_2 (split_word_pairs) → create_mapping_2 (build_char_map) ─┤
...                                                                     ─┤
extract_pairs_N (split_word_pairs) → create_mapping_N (build_char_map) ─┘
                                                                         ↓
                                           merge_mapping (merge_char_maps)
                                                                         ↓
                                         translate (decrypt_substitution)
```

- **`extract_pairs_i`** (one per example, **parallel**) — deterministic
  `split_word_pairs` splits encrypted and plain text into word pairs.
- **`create_mapping_i`** — deterministic `build_char_map` aligns chars.
- **`merge_mapping`** — deterministic `merge_char_maps` (majority vote).
- **`translate`** — deterministic `decrypt_substitution` applies the
  merged mapping, with vocabulary search for unmapped characters.

The number of parallel chains equals the number of example lines
(typically 5).  Tools are in `tools.py`.

### LLM-Generated DAGs (for retries and unknown types)

On solver failure (retry) or for unknown puzzle types, the `decompose` node
calls the LLM to generate a DAG structure. The LLM receives the puzzle type,
prompt, available tools, and failure context. It outputs a JSON array of
`ThoughtNode` objects deciding the decomposition strategy.

If the LLM fails to produce valid JSON, a single-node fallback DAG is
used (direct answer via LLM).

### Tool Set (grounded on train data analysis)

Every node uses a tool. `ask_llm` is the tool for reasoning; all others
are deterministic Python -- instant, 100% accurate. The `tool_input` field
supports `{parent_id}` placeholders so tool nodes can consume answers from
upstream nodes.

**General:**

| Tool            | Purpose                                  | Example input                              |
|-----------------|------------------------------------------|--------------------------------------------|
| `ask_llm`       | General LLM reasoning / extraction       | `{"question": "Which operation is consistent?"}` |
| `eval_math`     | Safe math expression                     | `{"expr": "0.5 * 9.8 * 3**2"}`            |
| `apply_formula` | Formula with named variables             | `{"formula": "0.5*g*t**2", "vars": {"g": 9.8, "t": 3}}` |
| `round_number`  | Round to n decimal places                | `{"value": 154.623, "decimals": 2}`       |
| `average`       | Mean of a list                           | `{"values": [15.88, 15.92]}`              |
| `regex_extract` | Extract regex matches (returns JSON)     | `{"text": "...", "pattern": "[\\d.]+"}`    |

**Bit manipulation** (all puzzles are 8-bit binary in/out):

| Tool           | Input                                     |
|----------------|-------------------------------------------|
| `xor_binary`   | `{"a": "10110010", "b": "01001101"}`      |
| `and_binary`   | `{"a": "10110010", "b": "01001101"}`      |
| `or_binary`    | `{"a": "10110010", "b": "01001101"}`      |
| `not_binary`   | `{"a": "10110010"}`                       |
| `shift_left`   | `{"a": "10110010", "n": 1, "bits": 8}`   |
| `shift_right`  | `{"a": "10110010", "n": 1, "bits": 8}`   |
| `rotate_left`  | `{"a": "10110010", "n": 1, "bits": 8}`   |
| `rotate_right` | `{"a": "10110010", "n": 1, "bits": 8}`   |

**Cipher / substitution:**

| Tool              | Purpose                                   |
|-------------------|-------------------------------------------|
| `build_char_map`       | Build substitution map from aligned pairs: `{"pairs": [["ucoov","queen"]]}` |
| `substitute_chars`     | Apply char mapping: `{"text": "ucoov", "mapping": {"u":"q",...}}` |
| `decrypt_substitution` | Decrypt ciphertext using map + vocab-guided permutation search for unmapped letters: `{"ciphertext": "trb", "mapping": {...}}` |

**Numeral conversion** (generalized: Roman + any base 2-36):

| Tool                    | Purpose                                         | Example                              |
|-------------------------|-------------------------------------------------|--------------------------------------|
| `to_roman`              | Integer to Roman numeral                        | `{"number": 38}` → `"XXXVIII"`      |
| `from_roman`            | Roman numeral to integer                        | `{"roman": "XXXVIII"}` → `"38"`     |
| `detect_numeral_system` | Try Roman + bases 2-36 against example pairs    | `{"pairs": [[38,"XXXVIII"],[15,"XV"]]}` → `{"system":"roman","all_correct":true,...}` |
| `convert_numeral`       | Convert decimal to any detected system          | `{"number": 42, "system": "base_2"}` → `"101010"` |

**Gravity physics** (d = 0.5gt^2):

| Tool                        | Purpose                                    |
|-----------------------------|--------------------------------------------|
| `compute_gravity_g`         | Compute g from observations: `{"observations": [[1.37, 14.92], ...]}` |
| `compute_gravity_d`         | Compute d from g and t: `{"g": 15.89, "t": 4.41}` -> `"154.62"` |
| `gravity_geom_mean_chain_exprs` | Parse `d_k =` mul/div chains from LLM text; return geometric mean (2dp) |

**Unit conversion:**

| Tool            | Purpose                                            |
|-----------------|----------------------------------------------------|
| `linear_factor` | Avg multiplicative factor from pairs: `{"pairs": [[10.08, 6.69], ...]}` |
| `linear_fit`    | Least-squares regression y=ax+b: `{"pairs": [[35.31,32.49],...]}` → `{"slope":a,"intercept":b}` |

## Threading Model

Each `solve_next` iteration:

1. **Find ready nodes**: nodes with `answer is None` whose parents all have answers.
2. **Submit to ThreadPoolExecutor**: one thread per ready node.
3. **Collect results**: `as_completed()` gathers answers as threads finish.
4. **On success**: write answers back, loop to find newly unblocked nodes.
5. **On failure**: record in `failure_log`, increment `retries`, route to `decompose`.

Typical parallelism is 1-3 threads per batch (e.g. `cipher_decryption` has
two root nodes that can run concurrently before `decrypt`).

## Retry Logic

When a node fails (LLM error, empty response, timeout), the graph routes
back to `decompose` with a `failure_log` containing `{node_id, question, error}`.

The decompose LLM receives the full failure context and is instructed to
generate a different decomposition. Suggested strategies (the LLM decides):

- Rephrase unclear sub-questions more explicitly
- Merge dependent steps that failed into fewer, simpler questions
- Try a completely different angle of attack
- Fall back to a single-node direct answer as last resort

Controlled by `MAX_RETRIES` (default: 3). If JSON parsing fails on any
attempt, a single-node fallback DAG is used automatically.

## State

```python
class ThoughtNode(TypedDict):
    id: str
    question: str
    depends_on: list[str]
    tool: Optional[str]        # deterministic tool name, or None for LLM
    tool_input: Optional[str]  # JSON string for tool params
    answer: Optional[str]

class FailureRecord(TypedDict):
    node_id: str
    question: str
    error: str

class GraphState(TypedDict):
    prompt: str
    puzzle_type: Optional[str]
    thought_dag: Optional[list[ThoughtNode]]
    retries: int
    failure_log: Optional[list[FailureRecord]]
    answer: Optional[str]
```

## Project Structure

```
Nemotron/
├── main.py                 # CLI entry point with --verbose DAG trace and --types filter
├── trace_row.py            # Optional: run one train id, print each DAG step I/O
├── requirements.txt        # Python dependencies
├── design.md               # This file
├── .env.example            # Template for config overrides
├── data/
│   ├── train.csv           # Kaggle train set (id, prompt, answer)
│   └── test.csv            # Kaggle test set  (id, prompt)
├── results/
│   └── predictions.csv     # Generated answers
└── src/
    ├── __init__.py
    ├── config.py            # Env vars: MODEL_NAME, OLLAMA_BASE_URL, LLM_PROVIDER, etc.
    ├── llm_client.py        # Unified LLM client (Ollama local / DeepSeek API fallback)
    ├── state.py             # ThoughtNode, FailureRecord, GraphState
    ├── classify.py          # Keyword classifier + deterministic DAG plan builder (all 6 types)
    ├── planner.py           # QueryPlanner: deterministic DAG builders for all puzzle types
    ├── decompose.py         # Pass-through on first pass; LLM re-decompose on retries
    ├── tools.py             # Deterministic tool functions (binary ops, math, substitution)
    ├── solver.py            # Threaded DAG solver; LLM nodes store full reply, sink answer = last line
    └── graph.py             # LangGraph wiring: classify -> decompose -> solve_next
```

## Configuration

| Variable          | Default                  | Description                                |
|-------------------|--------------------------|--------------------------------------------|
| `MODEL_NAME`      | `nemotron-3-nano:4b`     | Ollama model tag                           |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL                          |
| `LLM_PROVIDER`    | `auto`                   | `auto`, `ollama`, or `deepseek`            |
| `DEEPSEEK_API_KEY`| *(empty)*                | DeepSeek API key (required when provider≠ollama) |
| `DEEPSEEK_MODEL`  | `deepseek-chat`          | DeepSeek model name                        |
| `MAX_RETRIES`     | `3`                      | Max re-decompose attempts on failure       |
| `TRAIN_PATH`      | `data/train.csv`         | Path to training CSV                       |
| `TEST_PATH`       | `data/test.csv`          | Path to test CSV                           |
| `RESULTS_DIR`     | `results`                | Output directory                           |

## Model

**Primary: Nemotron 3 Nano** served locally via [Ollama](https://ollama.com/library/nemotron-3-nano).

- Default: `nemotron-3-nano:4b` (2.8 GB)
- Also available: `nemotron-3-nano` 30B MoE (24 GB)
- No API key required -- runs entirely locally.
- Uses native `ollama` Python client with `think=True` for chain-of-thought reasoning.

**Fallback: DeepSeek** (cloud API, OpenAI-compatible).

- Used when Ollama is unreachable and `LLM_PROVIDER=auto` (default), or
  when forced via `LLM_PROVIDER=deepseek`.
- Default model: `deepseek-chat`; set `DEEPSEEK_MODEL=deepseek-reasoner`
  for built-in chain-of-thought.
- Requires `DEEPSEEK_API_KEY` in `.env`.

All LLM calls go through the unified `LLMClient` (`src/llm_client.py`),
which dispatches to the selected backend.

- In **classify**, all six puzzle types use the `QueryPlanner` to build
  fully deterministic DAGs — zero LLM calls.  Deterministic solvers in
  `tools.py` handle the actual computation.

## Setup

1. **Install Ollama** and pull the model:

   ```bash
   ollama pull nemotron-3-nano:4b
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run (small sample with DAG trace)**:

   ```bash
   python main.py --limit 5 --verbose
   ```

4. **Run on test set**:

   ```bash
   python main.py --dataset data/test.csv --output results/submission.csv
   ```

## CLI

```bash
python main.py --help
```

| Flag        | Description                   | Default                   |
|-------------|-------------------------------|---------------------------|
| `--dataset` | Input CSV path                | `data/train.csv`          |
| `--output`  | Output CSV path               | `results/predictions.csv` |
| `--limit`   | Max rows to process           | all                       |
| `--verbose` | Print DAG execution trace     | off                       |
| `--types`   | Comma-separated puzzle types  | all types                 |

When the CSV has an `answer` column, printed **accuracy** treats two answers as
a match if strings are equal after strip, or if both parse as floats with
**absolute difference ≤ 10⁻²** (inclusive, plus a tiny epsilon for rounding).

## Discussion

### Expanding the tool set per puzzle type

Each puzzle type currently has a narrow set of tools tailored to the patterns
observed in the training data.  A natural next step is to broaden the tool
inventory for each category — for instance, adding more bitwise primitives
(population count, Gray code conversion, bit-field extraction), richer
numerical helpers (median, weighted average, polynomial fitting), or
domain-specific utilities (frequency analysis for ciphers, modular arithmetic
for equation transforms).  With a larger toolkit the LLM planner gains more
building blocks to compose solutions for question variants it has not seen
before, rather than being limited to the exact patterns the current tools were
designed for.

### Toward a type-agnostic, general-purpose runner

Today the pipeline classifies a puzzle first and then selects a type-specific
DAG plan.  A more ambitious direction is to make the tools themselves
general enough that the planner does not need to know the puzzle type at all.
Instead of `solve_gravity_physics` or `solve_cipher_decryption`, the system
would expose composable primitives — regex extraction, table lookup,
arithmetic evaluation, string substitution, statistical fitting, symbolic
equation solving — that apply across every category.  The planner would then
treat any incoming prompt the same way: read it, decide which primitives to
chain together, and execute the resulting DAG.  This would make the pipeline a
truly indiscriminate reasoning engine, able to generalise to new puzzle types
or mixed-domain questions without any classification step or type-specific
code paths.
