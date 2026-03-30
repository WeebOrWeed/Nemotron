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
classify   (keyword match -- no LLM, instant)
  │
  ▼
decompose  (LLM generates thought DAG as JSON)
  │
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
(no LLM inference, zero latency). Distribution is roughly even (~1,550
each in the training set).

| Type                 | Signature phrase                              | Task                                          |
|----------------------|-----------------------------------------------|-----------------------------------------------|
| `bit_manipulation`   | "bit manipulation"                            | Deduce bitwise op from examples, apply to new |
| `cipher_decryption`  | "encryption rules"                            | Build substitution map, decrypt target text   |
| `equation_transform` | "transformation rules"                        | Build symbol map, transform target equation   |
| `gravity_physics`    | "gravitational constant"                      | Regress g from d=0.5gt^2 examples, compute d  |
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

### LLM-Generated DAGs with Tool Dispatch

The `decompose` node calls the LLM to dynamically generate the DAG structure.
The LLM receives:

- The puzzle type (from classify)
- The full puzzle prompt
- The list of available deterministic tools with their input schemas
- On retry: the failure log with node IDs and error messages

The LLM outputs a JSON array of `ThoughtNode` objects. It decides:

- How many sub-steps to create (typically 2-6)
- Which nodes can run in parallel (independent `depends_on: []`)
- Which nodes need to wait for others (fan-out + merge patterns)
- **Whether each node should use a tool or the LLM**

The key design principle: **tools for computation, LLM for reasoning**.
The LLM is used for steps that require pattern recognition, interpretation,
or natural language understanding (e.g. "identify which bit operation is
consistent across examples"). Deterministic tools handle all computation
(e.g. actually XOR-ing two binary strings, evaluating `0.5 * g * t^2`,
applying a substitution map).

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

| Tool               | Purpose                                    |
|--------------------|--------------------------------------------|
| `compute_gravity_g`| Compute g from observations: `{"observations": [[1.37, 14.92], ...]}` |
| `compute_gravity_d`| Compute d from g and t: `{"g": 15.89, "t": 4.41}` -> `"154.62"` |

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

Typical parallelism is 1-3 threads per batch (e.g. `bit_manipulation`
fans out to 3 concurrent operations).

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
    ├── config.py            # Env vars: MODEL_NAME, OLLAMA_BASE_URL, MAX_RETRIES
    ├── state.py             # ThoughtNode, FailureRecord, GraphState
    ├── classify.py          # Deterministic keyword classifier (no LLM)
    ├── decompose.py         # LLM-generated DAG decomposition with tool assignment
    ├── tools.py             # Deterministic tool functions (binary ops, math, substitution)
    ├── solver.py            # Threaded DAG solver: tool dispatch or LLM fallback
    └── graph.py             # LangGraph wiring: classify -> decompose -> solve_next
```

## Configuration

| Variable          | Default                  | Description                                |
|-------------------|--------------------------|--------------------------------------------|
| `MODEL_NAME`      | `nemotron-3-nano:4b`     | Ollama model tag                           |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL                          |
| `MAX_RETRIES`     | `3`                      | Max re-decompose attempts on failure       |
| `TRAIN_PATH`      | `data/train.csv`         | Path to training CSV                       |
| `TEST_PATH`       | `data/test.csv`          | Path to test CSV                           |
| `RESULTS_DIR`     | `results`                | Output directory                           |

## Model

**Nemotron 3 Nano** served locally via [Ollama](https://ollama.com/library/nemotron-3-nano).

- Default: `nemotron-3-nano:4b` (2.8 GB)
- Also available: `nemotron-3-nano` 30B MoE (24 GB)
- No API key required -- runs entirely locally.
- Uses native `ollama` Python client with `think=False` for direct answers.

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
