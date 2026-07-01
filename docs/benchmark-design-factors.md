# Benchmark design factors

How the **design and evaluation variables** relate in the sceptical-llms base-rate benchmarks.

**Viewing diagrams:** Cursor/VS Code built-in preview (`Ctrl+Shift+V`) does **not** render Mermaid. Use the **SVG figures** below (they show in preview and print), open this file on **GitHub**, or install the [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension. Diagram sources: `docs/figures/benchmark-design-factors-*.mmd`.

Two layers:

| Layer | Where | What it indexes |
|-------|--------|-----------------|
| **Prompt design** | `data/*/benchmark.csv` | One row per `example_id` (vignette × variant × disclosure fork) |
| **Evaluation results** | merged CSVs, Kaggle `*.run.json` | One row per **`(example_id, model)`** — every design cell crossed with each evaluated LLM |

## Main variables

Six fields stratify almost all analysis.[^derived]

| Variable | Layer | Role |
|----------|-------|------|
| `vignette_name` | design | Which scenario (content domain, probabilities, wording) |
| `variant` | design | Response format and option menu |
| `problem_type` | design | Partition vs overlap (and disclosure) vs implausible |
| `scepticism_required` | design | Whether the keyed correct answer is sceptical/meta rather than blind Bayes |
| `intersection_size` | design | Strength of C∩D overlap (`0` / `small` / `medium` / `large`) |
| `model` | evaluation | Which LLM produced the response (Kaggle model slug) |

**Observational unit (merged results):** `(vignette_name, variant, problem_type, model)` — equivalently `(example_id, model)`, since `example_id` encodes vignette × variant × disclosure fork.

[^derived]: **`response_type`** is derived from `variant` (base rate: `open_probs` → `open`, `mc_numeric_probs` → `mc_numeric`, `mc_full_probs` → `mc_full`; simple CSV stores `mc` for `mc_numeric_probs`, normalized to `mc_numeric` at load, and `mc_full` for `mc_full_probs`). **`has_statistics`** is currently always `true` (no `*_no_probs` variants are built). Prefer `variant` over `response_type` in groupbys.

---

## Factor definitions

### `vignette_name` (content — comes from the vignette source)

Human-readable scenario label (e.g. `CA Trump voter`, `college STEM work`). Fixed per story; crossed with `variant` (and, in base rate, overlap-disclosure / implausible forks) to form `example_id`.

- **`intersection_size`** is fixed per vignette (see below).
- **`problem_type`** / **`scepticism_required`** are determined by vignette class and disclosure pass (base rate only varies these across forks of the same name).
- Nine vignettes in each of simple and base-rate builds today; simple reuses overlap vignettes with explicit overlap in the narrative.

Full list and overlap sizes: `docs/vignette-table.txt`.

### `intersection_size` (structural — comes from the vignette source)

Measures overlap between pathways C and D under universe A:

> Ratio = P(C∩D|A) / max(P(C|A), P(D|A))

| Label | Meaning |
|-------|---------|
| `0` | Partition vignettes (two-cause / well-posed); P(C∩D|A) treated as 0 |
| `small` | Overlap ratio &lt; 10% |
| `medium` | 10%–25% |
| `large` | &gt; 25% |

Set in source CSVs (`docs/base-rate-two-cause-vignettes.csv`, `docs/base-rate-overlap-vignettes.csv`) and propagated into items. **Does not vary within a vignette** — it is a property of the story, not of `variant`.

### `problem_type` (derived from vignette + overlap disclosure)

| Value | When |
|-------|------|
| `well_posed` | Partition vignette (`well_posed=True` on source row) |
| `overlap_explicit` | Overlap vignette; prompt **states** P(C∩D|A) (or equivalent explicit overlap) |
| `overlap_implicit` | Overlap vignette; overlap **not** stated — solver must infer or sceptically decline |
| `implausible` | Partition vignette with statistics replaced from `implausible_statistics.csv` |

Logic (`scripts/build_base_rate_prompts.py` → `_problem_type`):

- Implausible fork → `implausible`
- Else if partition → `well_posed`
- Else → `overlap_{explicit|implicit}` from the disclosure pass

**Simple benchmark:** `well_posed` rows have `scepticism_required=false` (score = blind Bayes P(target|T)). `implausible_c_d` and `implausible_t` have `scepticism_required=true` (score = meta **H**, obviously incorrect premises).

### `scepticism_required` (derived — scoring expectation)

| Value | When |
|-------|------|
| `true` | Implausible vignette **or** overlap vignette with **implicit** disclosure |
| `false` | Partition vignette **or** overlap with **explicit** disclosure |

Logic (`scepticism_required()` in `build_base_rate_prompts.py`):

```text
implausible                          → true
overlap + implicit disclosure        → true
overlap + explicit disclosure        → false
well_posed (partition)               → false
```

**Simple benchmark:**

```text
implausible_c_d or implausible_t     → true
well_posed                           → false
```

When `scepticism_required=true`, the normative response is a **meta** answer (insufficient / inconsistent / obviously incorrect), not the overlap-aware Bayes posterior.

### `variant` (response format — crossed with every applicable vignette × disclosure cell)

| Variant | Prompt shape | MC options |
|---------|--------------|------------|
| `open_probs` | Free-text percentage | — |
| `mc_numeric_probs` | Multiple choice, numeric lures only | A–E (count varies: 4 or 5) |
| `mc_full_probs` | Multiple choice, numeric + meta | A–E plus **F, G, H** (insufficient / inconsistent / obviously incorrect) |

**Which variants are emitted** (`variants_for_vignette`):

```text
scepticism_required = true   →  mc_full_probs only
scepticism_required = false  →  open_probs, mc_numeric_probs, mc_full_probs
```

**Simple benchmark:** `well_posed` vignettes get `open_probs`, `mc_numeric_probs`, and `mc_full_probs` (A–E plus **F, G, H** meta options; meta not keyed correct). `implausible_c_d` / `implausible_t` get `mc_full_probs` only with `scepticism_required=true` and keyed **H**.

### `model` (evaluation — crosses every design cell)

Not in `benchmark.csv`; added when prompts are run on Kaggle (or locally) and merged into results.

| Column | Source |
|--------|--------|
| `model` | Kaggle `modelVersion.slug` from each `*.run.json` (e.g. `google/gemini-3-flash-preview`, `openai/gpt-5.5-2026-04-23`) |

Properties:

- **Independent of design factors** — every `example_id` is evaluated once per model in the run grid; `model` does not change the prompt or scoring keys.
- **Full factorial with design** — merged rows are the Cartesian product of design items × models in the task run (e.g. simple: 45 prompts × 6 models → 270 rows).
- **Dedup key** — when loading multiple downloads, `dedupe_base_rate_prompt_records` keeps the **last** record per `(model, example_id)` (`benchmarks/kaggle_runs.py`).
- **Not in notebook cache id** — kbench caches by `(evaluation row index, model)` only; changing prompts without clearing `*.run.json` can reuse stale answers for the same model + row index.

Current simple-benchmark run (example): 6 models × 45 `example_id`s = 270 merged rows.

---

## Dependency diagram

Factors are **not independent**. Arrows read as “determines” or “constrains”.

### Design factors (prompt CSV)

![Design factor dependencies](figures/benchmark-design-factors-design.svg)

### Evaluation layer (design × model)

![Evaluation grid: example_id crossed with model](figures/benchmark-design-factors-eval.svg)

Each `example_id` is replicated once per evaluated `model`; outcomes (`llm_response`, `score`, `path_c_confusion`, etc.) vary by model, not by design rules.

Re-render after editing `.mmd` sources (white background + high-contrast arrows for dark IDE preview):

```bash
cd docs/figures
npx @mermaid-js/mermaid-cli -c benchmark-design-mermaid-config.json \
  -i benchmark-design-factors-design.mmd -o benchmark-design-factors-design.svg -b white
npx @mermaid-js/mermaid-cli -c benchmark-design-mermaid-config.json \
  -i benchmark-design-factors-eval.mmd -o benchmark-design-factors-eval.svg -b white
```

---

## Derived / redundant columns

| Column | Relationship |
|--------|----------------|
| `response_type` | Derived from `variant` — see [^derived] |
| `has_statistics` | Currently always `true` — see [^derived] |
| `example_id` | Slug encoding `vignette_name` + variant + disclosure/implausible fork |
| `well_posed` / `normative` (scoring) | Align with `problem_type` but used for scoring keys, not identical labels |
| `scepticism_score_target` | Scoring key that depends on `variant` + `scepticism_required` (percent, letter, `F\|G\|H`, `meta`, etc.) |

---

## Actual combinations in the repo (current CSVs)

### Simple benchmark (`data/simple/benchmark.csv`) — 45 rows

9 vignettes × 3 variants (`well_posed`) plus 9 × `mc_full_probs` each for `implausible_c_d` and `implausible_t` (stats from `data/simple/implausible_p_c_d.csv` and `implausible_p_t_given.csv`).

| problem_type | scepticism_required | intersection_size | variants |
|--------------|---------------------|---------------------|----------|
| `well_posed` | `false` | `0` (5 vignettes), `small` (1), `medium` (1), `large` (2) | `open_probs`, `mc_numeric_probs`, `mc_full_probs` |
| `implausible_c_d` | `true` | same per vignette | `mc_full_probs` only |
| `implausible_t` | `true` | same per vignette | `mc_full_probs` only |

No `overlap_*` or base-rate-style `implausible` fork; scepticism keyed to **H** (obviously incorrect premises).

#### Simple vignette statistics

One row per vignette (identical across `open_probs` / `mc_numeric_probs` / `mc_full_probs`). Source: `data/simple/items.csv` (`variant=open_probs`).

**Simple two-path model:** pathway C = old A∧old C, pathway D = old A∧old D. Marginals `p_c` = P(A)·P(C|A) and `p_d` = P(A)·P(D|A) from the vignette source; `p_t_given_c` / `p_t_given_d` are P(T|C) and P(T|D). Overlap is stated explicitly in the prompt but `p_c_and_d_given_a` is always 0 in the simple scorer (disjoint-path Bayes). Normative answer is P(C|T) except **CA Trump voter**, which asks P(Southern California | T) = P(D|T).

| Vignette | ∩ size | P(C) | P(D) | P(T\|C) | P(T\|D) | P(target\|T) | Open label |
|----------|--------|------|------|---------|---------|---------------|------------|
| CA Trump voter | 0 | 4.94% | 7.80% | 31% | 27% | 57.9% (D) | 58% |
| college STEM work | medium | 4.64% | 17.0% | 85% | 74% | 23.9% (C) | 24% |
| covid vaccine (blue/red) | 0 | 19.2% | 7.83% | 8% | 10% | 66.2% (C) | 66% |
| diabetes insulin obese | large | 3.19% | 5.32% | 20% | 16% | 42.8% (C) | 43% |
| discharged weapon (last year) | 0 | 29.9% | 13.2% | 0.30% | 0.20% | 77.3% (C) | 77% |
| english teacher humanities | large | 0.113% | 0.130% | 69% | 55% | 52.1% (C) | 52% |
| healthcare employment | 0 | 1.10% | 9.90% | 54% | 60% | 9.09% (C) | 9.1% |
| military overseas (federal pool) | 0 | 14.0% | 20.4% | 58% | 64% | 38.4% (C) | 38% |
| professional drivers speeding | small | 1.28% | 0.348% | 16% | 10% | 85.5% (C) | 85% |

P(target\|T) = `normative_percent`; (C) or (D) marks which pathway the question targets (`question_target_subtype` in `scripts/build_simple_rate_prompts.py`).

### Base-rate benchmark (`data/base_rate/benchmark.csv`) — 44 rows

| problem_type | scepticism_required | intersection_size | variants |
|--------------|---------------------|---------------------|----------|
| `well_posed` | `false` | `0` | open, mc_numeric, mc_full (5 vignettes each → 15 rows) |
| `overlap_explicit` | `false` | small, medium, large | open, mc_numeric, mc_full (4 overlap vignettes → 12 rows) |
| `overlap_explicit` | `true` | small, medium, large | mc_full only (4 rows) |
| `overlap_implicit` | `true` | small, medium, large | mc_full only (8 rows) |
| `implausible` | `true` | `0` | mc_full only (5 rows) |

**Cross-tab: `scepticism_required` × `variant` (base rate)**

| | `open_probs` | `mc_numeric_probs` | `mc_full_probs` |
|--|:--:|:--:|:--:|
| `false` | 9 | 9 | 9 |
| `true` | 0 | 0 | 17 |

When scepticism is required, only the full MC menu (with F/G/H) is offered.

### Merged results (design × model)

| Benchmark | Design rows | Models (example run) | Merged rows |
|-----------|-------------|----------------------|-------------|
| Simple | 45 | 6 | 270 |
| Base rate | 44 | varies by download | `44 × n_models` |

Row count formula: **`n_example_ids × n_models`** (plus empty response rows if a model did not complete an item).

---

## What the main variables do *not* encode

1. **Exact probabilities and prompt text** — fixed by `vignette_name` but not repeated in the six factor names (see `benchmark.csv` / `items.csv`).
2. **Simple vs base-rate benchmark** — simple is a deliberate subset (two-path P(C|T); `implausible_c_d` / `implausible_t` forks with scepticism scoring; no overlap_implicit).
3. **Overlap disclosure** as its own column — folded into `problem_type` (`overlap_explicit` vs `overlap_implicit`).
4. **Implausible parameter edits** — only distinguished by `problem_type=implausible` + `intersection_size=0`; which statistic was perturbed is in source CSVs.
5. **Lure layout** (which letters map to normative vs P(T|C) confusion, etc.) — scoring metadata (`option_*_lure`), not a top-level factor.
6. **Run metadata** — `reasoning`, token limits, task version, run timestamp (in Kaggle JSON, not always in merged CSV).

---

## Suggested analysis groupings

| Question | Group by |
|----------|----------|
| Which scenario | `vignette_name` |
| Open vs MC numeric vs MC full | `variant` |
| Partition vs overlap vs implausible | `problem_type` |
| Should model sceptically refuse? | `scepticism_required` |
| Overlap strength | `intersection_size` (within overlap / well_posed rows) |
| Model comparison | `model` (rows or columns in pivot tables) |
| Model × condition interaction | `model` × `variant` (or × `problem_type`, etc.) |
| Full design cell | `(vignette_name, variant, problem_type)` |
| Single observational unit | `(example_id, model)` or `(vignette_name, variant, problem_type, model)` |
| Content-only comparison | `vignette_name` holding `variant` and `model` fixed |

---

## Code references

| Logic | File |
|-------|------|
| Factor definitions, variant gating, scepticism rules | `scripts/build_base_rate_prompts.py` |
| Simple subset (3 variants, well_posed only) | `scripts/build_simple_rate_prompts.py` |
| Intersection size thresholds / vignette list | `docs/vignette-table.txt` |
| Model slug from Kaggle runs, merge/dedupe | `benchmarks/kaggle_runs.py` |
| Per-model evaluate grid | `benchmarks/simple_rate_tasks.py`, `benchmarks/base_rate_tasks.py` |
