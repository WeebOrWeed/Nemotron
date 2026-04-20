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
classify   (keyword match + QueryPlanner.plan(puzzle_type, prompt) via unified LLM planner)
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
| `bit_manipulation`   | "bit manipulation"                            | LLM plans DAG → `solve_bit_manipulation` (brute-force per-bit boolean function search) |
| `cipher_decryption`  | "encryption rules"                            | LLM plans DAG → N parallel [split_word_pairs → build_char_map] → merge_char_maps → decrypt_substitution |
| `equation_transform` | "transformation rules"                        | LLM plans DAG → `solve_equation_transform` (operator-centric compound-operation search) |
| `gravity_physics`    | "gravitational constant"                      | LLM plans DAG → extract_gravity_obs → compute_gravity_g → compute_gravity_d |
| `numeral_conversion` | "numeral system"                              | LLM plans DAG → `solve_numeral_conversion` (detect Roman numerals, convert) |
| `unit_conversion`    | "unit conversion"                             | LLM plans DAG → extract_unit_pairs → geometric_mean_factor → apply_factor_round |

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
  expands to `answer["field"]` (e.g. `{extract_obs_target_t}` from `{"observations":[...],"target_t":4.41}`).

### Execution Strategy

DAG execution uses **composable deterministic tools** — small, focused
functions that each handle one step.  A single unified LLM planner
generates the mermaid DAG topology at plan time; the puzzle type is
passed as an intent parameter so the LLM selects the correct plan.

| Type                 | DAG Structure                                                       |
|----------------------|---------------------------------------------------------------------|
| `gravity_physics`    | 3 nodes: `extract_gravity_obs` → `compute_gravity_g` (weighted LS) → `compute_gravity_d` |
| `unit_conversion`    | 3 nodes: `extract_unit_pairs` → `geometric_mean_factor` → `apply_factor_round` |
| `numeral_conversion` | Single node: `solve_numeral_conversion` — detect Roman numerals, convert target |
| `cipher_decryption`  | N parallel [split_word_pairs → build_char_map] → merge_char_maps → decrypt_substitution |
| `bit_manipulation`   | Single node: `solve_bit_manipulation` — brute-force per-bit boolean function search |
| `equation_transform` | Single node: `solve_equation_transform` — operator-centric compound-operation search |

If a solver node fails, the graph routes back to `decompose` for LLM retry.

### QueryPlanner (unified LLM-based planner)

All six puzzle types go through a **single** `QueryPlanner.plan(puzzle_type, prompt)`
method in `src/planner.py`:

1. One unified `PLANNER_SYSTEM` prompt describes all six types, their
   composable tools, and the expected DAG topology.
2. The caller passes `puzzle_type` as the intent; the LLM reads
   `PUZZLE_TYPE: <type>` in the user message and generates the matching DAG.
3. The LLM outputs MERMAID edges + JSON node definitions.
4. A single generic `_build_dag` builder converts the parsed output into
   `ThoughtNode` objects — `depends_on` is derived from mermaid edges,
   and `"__PROMPT__"` placeholders in `tool_input` are substituted with
   the actual prompt text (JSON-escaped).

No per-type hardcoded builders — the LLM output is trusted directly.

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

**Gravity physics** (d = 0.5gt²):

| Tool                    | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `extract_gravity_obs`   | Extract (t,d) pairs and target_t from prompt: `{"prompt": "..."}` → `{"observations": [...], "target_t": 4.41}` |
| `compute_gravity_g`     | Weighted least-squares g from observations: `{"observations": [[1.37, 14.92], ...]}` → `"15.89"` |
| `compute_gravity_d`     | Compute d = 0.5·g·t² with ceil/floor rounding: `{"g": "15.89", "t": "4.41"}` → `"154.62"` |

**Unit conversion:**

| Tool                    | Purpose                                                              |
|-------------------------|----------------------------------------------------------------------|
| `extract_unit_pairs`    | Extract (from, to) pairs and target from prompt: `{"prompt": "..."}` → `{"pairs": [...], "target": 25.09}` |
| `geometric_mean_factor` | Geometric mean of y/x ratios: `{"pairs": [[10.08, 6.69], ...]}` → `"0.6636..."` |
| `apply_factor_round`    | Factor × target with ceil/floor rounding: `{"factor": "0.6636", "target": "25.09"}` → `"16.65"` |

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
├── trace_row.py            # Run one train id, print each DAG step I/O
├── train_planner.py        # RL scoring harness: N candidate DAGs per puzzle → JSONL
├── requirements.txt        # Python dependencies
├── design.md               # This file
├── .env.example            # Template for config overrides
├── notebooks/
│   ├── grpo_planner.ipynb  # All-in-one Kaggle RL notebook (collect + train + output)
│   └── kernel-metadata.json # Kaggle kernel push metadata
├── kaggle_upload/
│   └── nemotron-code/      # Project source packaged as Kaggle dataset
│       ├── dataset-metadata.json
│       ├── src/             # Mirror of src/ for Kaggle
│       └── data/train.csv
├── data/
│   ├── train.csv           # Kaggle train set (id, prompt, answer)
│   ├── test.csv            # Kaggle test set  (id, prompt)
│   ├── gravity_20.csv      # Optional: first 20 gravity_physics rows from train
│   ├── unit_20.csv         # Optional: first 20 unit_conversion rows from train
│   └── planner_scores.jsonl # RL harness output: scored DAG candidates
├── results/
│   ├── predictions.csv     # Generated answers (default main.py output)
│   └── unit_20_predictions.csv  # Example batch: main.py --dataset data/unit_20.csv
└── src/
    ├── __init__.py
    ├── config.py            # Env vars: MODEL_NAME, OLLAMA_BASE_URL, LLM_PROVIDER, etc.
    ├── llm_client.py        # Unified LLM client (Ollama local / DeepSeek API fallback)
    ├── state.py             # ThoughtNode, FailureRecord, GraphState
    ├── classify.py          # Keyword classifier + LLM DAG planner (OpenRouter / DeepSeek)
    ├── planner.py           # QueryPlanner: LLM composes DAGs from tool catalogue
    ├── decompose.py         # Pass-through on first pass; LLM re-decompose on retries
    ├── tools.py             # Deterministic tool functions (binary ops, math, substitution)
    ├── tools_equation.py    # equation_transform solver (extracted for independent work)
    ├── solver.py            # Threaded DAG solver; LLM nodes store full reply, sink answer = last line
    └── graph.py             # LangGraph wiring: classify -> decompose -> solve_next
```

## Configuration

| Variable              | Default                                    | Description                                |
|-----------------------|--------------------------------------------|--------------------------------------------|
| `MODEL_NAME`          | `nemotron-3-nano:4b`                       | Ollama model tag                           |
| `OLLAMA_BASE_URL`     | `http://localhost:11434`                   | Ollama server URL                          |
| `LLM_PROVIDER`        | `auto`                                     | `auto`, `ollama`, `openrouter`, or `deepseek` |
| `OPEN_ROUTER_API_KEY` | *(empty)*                                  | OpenRouter API key                         |
| `OPEN_ROUTER_MODEL`   | `nvidia/llama-3.1-nemotron-70b-instruct`   | OpenRouter model name                      |
| `DEEPSEEK_API_KEY`    | *(empty)*                                  | DeepSeek API key                           |
| `DEEPSEEK_MODEL`      | `deepseek-chat`                            | DeepSeek model name                        |
| `MAX_RETRIES`         | `3`                                        | Max re-decompose attempts on failure       |
| `TRAIN_PATH`          | `data/train.csv`                           | Path to training CSV                       |
| `TEST_PATH`           | `data/test.csv`                            | Path to test CSV                           |
| `RESULTS_DIR`         | `results`                                  | Output directory                           |

## Model

**Primary: Nemotron 3 Nano** served locally via [Ollama](https://ollama.com/library/nemotron-3-nano).

- Default: `nemotron-3-nano:4b` (2.8 GB)
- Also available: `nemotron-3-nano` 30B MoE (24 GB)
- No API key required -- runs entirely locally.
- Uses native `ollama` Python client with `think=True` for chain-of-thought reasoning.

**Fallback 1: OpenRouter** (cloud API, OpenAI-compatible).

- Used when Ollama is unreachable and `OPEN_ROUTER_API_KEY` is set
  (with `LLM_PROVIDER=auto`), or when forced via `LLM_PROVIDER=openrouter`.
- Default model: `nvidia/llama-3.1-nemotron-70b-instruct`; override with
  `OPEN_ROUTER_MODEL`.
- Requires `OPEN_ROUTER_API_KEY` in `.env`.

**Fallback 2: DeepSeek** (cloud API, OpenAI-compatible).

- Used when Ollama and OpenRouter are both unavailable, or when forced
  via `LLM_PROVIDER=deepseek`.
- Default model: `deepseek-chat`; set `DEEPSEEK_MODEL=deepseek-reasoner`
  for built-in chain-of-thought.
- Requires `DEEPSEEK_API_KEY` in `.env`.

All LLM calls go through the unified `LLMClient` (`src/llm_client.py`),
which dispatches to the selected backend. Auto-resolution order:
Ollama → OpenRouter → DeepSeek.

- In **classify**, a dedicated cloud LLMClient (OpenRouter when
  `OPEN_ROUTER_API_KEY` is set, otherwise DeepSeek when
  `DEEPSEEK_API_KEY` is set) powers the `QueryPlanner` which composes
  DAGs from the tool catalogue.  Execution nodes use the configured
  `LLM_PROVIDER`.

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

## RL Training Loop for the DAG Planner

The planner LLM can be improved via reinforcement learning.
`train_planner.py` is the scoring harness that collects training data.

```
train_planner.py
  |
  |-- Loads train.csv
  |-- For each puzzle:
  |     1. Planner generates N candidate DAGs (temperature sampling)
  |     2. Each DAG is executed through the real solver
  |     3. Compare final answer to ground truth -> reward score
  |     4. Log (puzzle_id, planner_output, reward, ...) to JSONL
  |
  |-- The scored JSONL can feed:
  |     a) Few-shot prompt optimization (add best DAGs to PLANNER_SYSTEM)
  |     b) GRPO fine-tuning (with TRL / OpenRLHF on a local model)
  |     c) Tool-construction agent (analyze failures, propose new tools)
```

Reward function:

| Outcome           | Reward |
|-------------------|--------|
| Exact match       | +1.0   |
| Numeric within 0.01 | +0.5 |
| Valid DAG, wrong answer | -0.5 |
| Malformed DAG     | -1.0   |

Usage:

```bash
python train_planner.py --n 8 --limit 200 --types gravity_physics
python train_planner.py --n 4 --limit 50   # all types, 4 candidates each
```

Output goes to `data/planner_scores.jsonl` by default (one JSON object per
candidate per puzzle).

### Kaggle RL Pipeline

The full RL loop (data collection + failure analysis + training) can run
on Kaggle using `notebooks/grpo_planner.ipynb`. The notebook:

1. **Collects** scored data by calling DeepSeek as the planner LLM and
   executing DAGs via the uploaded `src/tools.py`.  Each node's execution
   trace (tool, input, output, error) is captured for analysis.
2. **Analyzes failures** -- for every VALID-but-wrong sample, DeepSeek
   classifies the root cause (bad plan, missing tool, tool bug, bad input)
   and proposes a fix.  For `NEW_TOOL` proposals, it generates concrete
   Python function code ready to paste into `tools.py`.
3. **Trains** a QLoRA adapter on Qwen-7B using SFTTrainer on the winning
   plans.
4. **Outputs** the LoRA adapter, an updated `PLANNER_SYSTEM` with embedded
   few-shot examples, and a `failure_analysis.json` with all proposals and
   generated tool code.

Supporting files:

| File | Purpose |
|------|---------|
| `notebooks/grpo_planner.ipynb` | All-in-one Kaggle notebook |
| `notebooks/kernel-metadata.json` | Kaggle kernel metadata (GPU, Internet, dataset link) |
| `kaggle_upload/nemotron-code/` | Project source packaged as a Kaggle dataset |

Sync artifacts back:

```bash
kaggle kernels output tianlinzhao1/grpo-dag-planner-training -p kaggle_output/
# Copy planner_system_updated.txt content into PLANNER_SYSTEM in src/planner.py
```

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
Instead of monolithic end-to-end solvers, the system
would expose composable primitives — regex extraction, table lookup,
arithmetic evaluation, string substitution, statistical fitting, symbolic
equation solving — that apply across every category.  The planner would then
treat any incoming prompt the same way: read it, decide which primitives to
chain together, and execute the resulting DAG.  This would make the pipeline a
truly indiscriminate reasoning engine, able to generalise to new puzzle types
or mixed-domain questions without any classification step or type-specific
code paths.
