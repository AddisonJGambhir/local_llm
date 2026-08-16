# Qwen3.6 model-quality benchmark plan

This directory is reserved for a reproducible quality benchmark comparing the
four locally installed Qwen3.6 GGUF models. It currently contains planning
documentation only. No benchmark harness has been implemented or executed.

## Models under test

| ID | Local GGUF |
|---|---|
| `27b-iq4` | `models/Qwen3.6-27B-IQ4_NL.gguf` |
| `27b-q4km` | `models/Qwen3.6-27B-MTP-Q4_K_M.gguf` |
| `35b-iq4` | `models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` |
| `35b-q4km` | `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |

The benchmark must answer two distinct questions:

1. How much quality does IQ4_NL lose relative to Q4_K_M within each
   architecture?
2. How do the dense 27B and MoE 35B-A3B models compare at each quantization?

These effects must be reported separately. A single aggregate ranking is not
enough.

## Fairness controls

All quality runs must use:

- One pinned llama.cpp revision and backend.
- The same chat-template implementation and prompt serialization.
- Identical context, generation, tool, and wall-clock budgets.
- Identical Q8_0 K/V cache types unless a benchmark requires otherwise.
- One request at a time with no competing GPU workload.
- A fresh server process and empty prompt cache for each model.
- No MTP, DFlash, or other speculative decoding.
- No result-dependent retries.
- Exact model, tokenizer, template, binary, dataset, and configuration hashes
  recorded in the run manifest.

The 27B Q4_K_M artifact includes MTP tensors while the 27B IQ4_NL artifact does
not. Those tensors must remain unused. Otherwise the experiment would mix
weight quality with a decoding optimization.

Performance measurements such as latency and tokens per second may be recorded,
but they must not affect quality scores.

## Benchmark battery

### Quantization fidelity

Evaluate each IQ4_NL model against its same-architecture Q4_K_M model on a fixed
corpus with balanced natural-language, source-code, and mathematical text.

Record separately for each corpus category:

- Perplexity and relative perplexity increase.
- Mean and percentile KL divergence.
- Correct-token probability change.
- RMS token-probability error.
- Top-token agreement.

Required comparisons:

- `27b-iq4` versus `27b-q4km`
- `35b-iq4` versus `35b-q4km`

Do not calculate cross-architecture KL divergence. The 27B and 35B models have
different underlying weights, so such a value would not represent quantization
loss.

### Code generation

| Suite | Scope | Primary metric |
|---|---:|---|
| HumanEval+ | Full suite | pass@1 |
| MBPP+ | Full suite | pass@1 |
| LiveCodeBench v6 | Full suite | pass@1 |

LiveCodeBench results should also be split by problem difficulty, publication
date, and failure type. Code must be executed in isolated containers with
resource and network limits.

### Repository-level coding

Use SWE-bench Verified with one frozen agent scaffold for every model.

The scaffold must provide exactly the same:

- System prompt.
- Repository state.
- Bash, search, and file-editing tools.
- Context and generated-token limits.
- Turn and tool-call limits.
- Docker images.
- Network policy.
- Per-task timeout.

Execution stages:

1. Validate the harness using gold patches.
2. Run a ten-task smoke test.
3. Run a stratified 100-task pilot.
4. Run all 500 SWE-bench Verified tasks.
5. Repeat model-disagreement tasks with additional fixed seeds.

Primary metric: percentage of instances resolved by the official Docker
evaluation harness. Also report tool errors, invalid patches, timeouts, token
usage, turns, and test-regression categories.

Terminal-Bench 2.0 may be added as a separate terminal-agent domain. Its score
must not be merged into SWE-bench.

### Mathematics and reasoning

| Suite | Scope | Primary metric |
|---|---:|---|
| MATH-500 | Full suite | exact-answer accuracy |
| AIME 2026 I and II | 30 problems, multiple seeds | pass@1 and majority vote |
| GPQA Diamond | Full suite, multiple seeds | accuracy |
| MMLU-Pro | All subjects | macro and per-subject accuracy |

Answer extraction must be deterministic and tested against known examples.
Malformed and missing answers count as failures.

### Instruction following

Run the full IFEval suite and report:

- Strict prompt accuracy.
- Strict instruction accuracy.
- Loose prompt accuracy.
- Loose instruction accuracy.

IFEval should use the explicitly documented non-thinking configuration. It is
intended to measure instruction compliance rather than hidden reasoning length.

### Long context

Run all RULER task categories at:

- 8K tokens
- 32K tokens
- 128K tokens
- 256K tokens

Use deterministic decoding. Report per-task accuracy at every length and an
accuracy-versus-context curve. The 256K run is a local extension and must be
labeled separately from official RULER comparisons.

## Generation configurations

Use Qwen's documented sampling parameters as the primary sampled evaluation:

### Precise coding

```text
temperature=0.6
top_p=0.95
top_k=20
min_p=0.0
presence_penalty=0.0
repeat_penalty=1.0
```

### General reasoning

```text
temperature=1.0
top_p=0.95
top_k=20
min_p=0.0
repeat_penalty=1.0
```

Qwen documents different general-task presence penalties for the two
architectures. Therefore general reasoning should have two reported lanes:

1. **Official settings:** each architecture uses its documented parameter.
2. **Controlled settings:** every model uses the same presence penalty.

Greedy decoding may be used for logit fidelity, constrained multiple-choice,
answer-extraction validation, and RULER. It should not replace the primary
sampled coding and reasoning evaluation.

Every sampled task must use a predetermined seed schedule shared by all models.

## Statistical analysis

Report raw scores before any aggregate score. For every applicable suite,
calculate:

- Paired bootstrap 95% confidence intervals.
- Pairwise win/loss/tie counts.
- McNemar tests for paired binary outcomes.
- Holm correction across model-pair comparisons.
- Between-seed variance.
- Invalid-output, truncation, timeout, and repetition rates.

Primary contrasts:

```text
27b-q4km - 27b-iq4
35b-q4km - 35b-iq4
27b-q4km - 35b-q4km
27b-iq4  - 35b-iq4
```

Use a task-level mixed-effects analysis where practical:

```text
success ~ architecture + quantization + architecture:quantization + task
```

This estimates whether IQ4_NL affects the dense and MoE architectures
differently.

## Aggregate reporting

The final report must contain:

- Per-suite raw scores and confidence intervals.
- Quantization deltas within each architecture.
- Architecture deltas within each quantization.
- Pairwise task-disagreement lists.
- Failure examples grouped by cause.
- Seed reliability.
- Long-context accuracy curves.
- Agent token and tool budgets consumed.
- Latency and throughput in a separate performance section.

If an overall score is presented, first normalize within each suite and then
weight domains equally. Do not weight individual questions equally across the
entire battery, because large suites would dominate the result.

Suggested top-level domains:

1. Code generation.
2. Repository/agent coding.
3. Mathematics.
4. Scientific and broad reasoning.
5. Instruction following.
6. Long-context use.
7. Quantization fidelity.

## Planned run layout

Future implementation should keep immutable inputs separate from generated
results:

```text
tests/model_quality/
├── README.md
├── manifests/
├── prompts/
├── scorers/
└── harness/

output/model-quality/<run-id>/
├── manifest.json
├── environment.json
├── generations/
├── scores/
├── logs/
└── report/
```

The output directory must include enough information to reproduce every
generation and score without depending on mutable defaults.

## Acceptance criteria

The benchmark is ready for a full run only when:

- Model and tokenizer identities are verified.
- Prompt token IDs match where expected.
- MTP and all speculative decoding are demonstrably disabled.
- Gold-patch SWE-bench validation succeeds.
- Code-execution sandboxes reject network and filesystem escape attempts.
- Every scorer passes known-answer unit tests.
- Interrupted runs resume without silently duplicating samples.
- A smoke run produces complete manifests, generations, scores, and logs.
- Repeated deterministic tasks produce identical answers.

## Expected resource requirements

Plan for:

- At least 250 GB of free workspace for Docker images, datasets, generations,
  and temporary logits.
- Several hours for smoke and scorer validation.
- One to three days for the objective core battery.
- Additional days for RULER.
- Several days to weeks for full SWE-bench Verified, depending on agent
  trajectories and retry policy.

Dataset versions, repository commits, container digests, and evaluation-tool
versions must be pinned before the first scored run.
