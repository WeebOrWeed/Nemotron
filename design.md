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
classify   (keyword match + build recommended DAG with tools -- no LLM, instant)
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
| `bit_manipulation`   | "bit manipulation"                            | Deduce bitwise op from examples, apply to new |
| `cipher_decryption`  | "encryption rules"                            | Build substitution map, decrypt target text   |
| `equation_transform` | "transformation rules"                        | Build symbol map, transform target equation   |
| `gravity_physics`    | "gravitational constant"                      | Expand t*t; g chains; LLM emits d_i * / lines; Python tool geometric mean (2dp) |
| `numeral_conversion` | "numeral system"                              | Identify base from examples, convert target   |
| `unit_conversion`    | "unit conversion"                             | Compute linear factor from examples, convert  |

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

### Execution Strategy: LLM-per-Step + Deterministic Fallback

Each puzzle type is solved by breaking the problem into focused LLM
sub-steps (the DAG-of-Thoughts approach). This demonstrates reasoning
technique improvement using Nemotron, which is the goal of the challenge.
Deterministic solvers are kept as fallbacks on retry.

**LLM-per-step (primary path):**

| Type                 | DAG Nodes (all LLM)                                                |
|----------------------|--------------------------------------------------------------------|
| `gravity_physics`    | extract → fill → **expand** → **g = d/0.5/t/t** → symbolic d_i → **LLM: d_k chain lines only** → **tool** `gravity_geom_mean_chain_exprs` (geometric mean ∏d_i^{1/n}, 2dp) |
| `unit_conversion`    | extract [from,to] pairs + target → avg factor to/from → apply × target |
| `numeral_conversion` | extract examples JSON + target decimal → infer system → express target |
| `cipher_decryption`  | extract word pairs ∥ extract ciphertext → decrypt (substitution)   |
| `bit_manipulation`   | infer rule from all I/O pairs → apply rule to target 8-bit input   |
| `equation_transform` | extract examples JSON + 5-char target → infer rule → apply to target |

Deterministic helpers (`solve_equation_transform`, `solve_bit_manipulation`,
etc.) remain in `tools.py`. If `decompose` emits `solve_equation_transform`
and it fails, `solver` still uses an LLM majority-vote fallback for that node.

> **Migration in progress:** All six puzzle types use LLM-per-step DAGs in
> `classify`. End-to-end solvers in `tools.py` support retries, optional tool
> nodes from `decompose`, and targeted fallbacks in `solver`.

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
| `build_char_map`  | Build substitution map from aligned pairs: `{"pairs": [["ucoov","queen"]]}` |
| `substitute_chars`| Apply char mapping: `{"text": "ucoov", "mapping": {"u":"q",...}}` |

**Numeral conversion** (100% Roman numerals in dataset):

| Tool         | Example                            | Result       |
|--------------|------------------------------------|--------------|
| `to_roman`   | `{"number": 38}`                   | `"XXXVIII"`  |
| `from_roman` | `{"roman": "XXXVIII"}`             | `"38"`       |

**Gravity physics** (d = 0.5gt^2):

| Tool                        | Purpose                                    |
|-----------------------------|--------------------------------------------|
| `compute_gravity_g`         | Compute g from observations: `{"observations": [[1.37, 14.92], ...]}` |
| `compute_gravity_d`         | Compute d from g and t: `{"g": 15.89, "t": 4.41}` -> `"154.62"` |
| `gravity_geom_mean_chain_exprs` | Parse `d_k =` mul/div chains from LLM text; return geometric mean (2dp) |

**Unit conversion:**

| Tool            | Purpose                                            |
|-----------------|----------------------------------------------------|
| `linear_factor` | Avg factor from pairs: `{"pairs": [[10.08, 6.69], ...]}` |

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
├── main.py                 # CLI entry point with --verbose DAG trace
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
    ├── classify.py          # Keyword classifier + DAG plan builder (no LLM)
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

- In **classify**, all six puzzle types use LLM-per-step DAGs; deterministic
  solvers in `tools.py` are used when `decompose` emits them or for targeted
  fallbacks (e.g. failed `solve_equation_transform` node).

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

When the CSV has an `answer` column, printed **accuracy** treats two answers as
a match if strings are equal after strip, or if both parse as floats with
**absolute difference ≤ 10⁻²** (inclusive, plus a tiny epsilon for rounding).
