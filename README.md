# proj-jailbreak-sandbox

Artifact for the paper **"Exploring and Developing a Pre-Model Safeguard with Draft Models"**
by Hongyu Cai (Purdue), Arjun Arunasalam (FIU), Yiming Liang (Purdue),
Antonio Bianchi (Purdue), and Z. Berkay Celik (Purdue).

This repository contains the source code, datasets, and scripts to reproduce
the main experimental results of the paper. The artifact covers:

1. A systematic study of jailbreak transferability from large target LLMs
   to small draft language models (SLMs).
2. The proposed **Sandbox** safeguard, which uses speculative draft-model
   inference combined with existing guards to detect jailbreak prompts
   *before* invoking the target LLM.
3. Effectiveness (ASR / DSR / FNR) and efficiency (latency) measurements
   on both malicious (RPAB) and benign (Just-Eval) prompts.

> **Online repository:** https://github.com/standbyme/proj-jailbreak-sandbox

> **Notice:** This repository, by its research nature, contains examples of
> prompts that are designed to elicit harmful or unaligned responses from
> LLMs. The provided code is intended *exclusively* for academic research on
> LLM safety and defense. Do not deploy in production or against systems for
> which you do not have authorization.

---

## 1. Hardware and software requirements

**Hardware (recommended).** All GPU-bound steps were originally executed on
SLURM A100 (80 GB) nodes with `--gpus-per-node=1` and 32 GB of host RAM.
The minimum hardware to reproduce the *full* matrix is therefore a single
A100-80GB-class GPU, because the largest target models
(Llama-3-70B-Instruct-AWQ, Qwen-1.5-72B-Chat-AWQ) cannot fit on smaller
cards even after AWQ quantization. Smaller scale runs (one generation
method, the `mini_RPAB` dataset, draft-only experiments) work on a single
A100-40GB or two consumer 24 GB GPUs.

**Software.**

- Linux (tested on RHEL 8 / Ubuntu 22.04) with NVIDIA driver supporting
  CUDA 12.x. CUDA 12.6 is the version we used in production.
- Python 3.12 (pinned in `.python-version`).
- [`uv`](https://github.com/astral-sh/uv) ≥ 0.9 for dependency management.
- (SLURM-only) the cluster module system providing `cuda/12.6.0`.
- (Optional) An OpenAI API key for the GPT-judge evaluation (step 4) and
  for the `GPTFuzzer` / `PAIR` attackers that rely on a chat model
  (`gpt-5-nano-2025-08-07` is used by default).

**Disk.** A complete reproduction with the full RPAB dataset, all six
draft models, all three target models, all five attackers, and all guards
produces ≈80 GB of intermediate pickles under `workdir/`.

## 2. Repository layout

```
.
├── pyproject.toml              # Python project + dependency manifest (uv)
├── uv.lock                     # Pinned lockfile
├── .python-version             # 3.12
├── src/
│   ├── core/                   # Pipeline scripts (step_1 ... step_13)
│   │                           # plus shared modules: language_model.py,
│   │                           # guard.py, evaluation.py, multifaceted.py,
│   │                           # ppl_calculator.py, utils.py, ...
│   └── slurm/                  # SLURM job templates (*.tpl) and matrix
│                               # files (*.yaml) for each step
└── workdir/                    # Runtime working directory
    ├── dataset/
    │   ├── RPAB.jsonl          # 50 malicious intents (full set)
    │   └── mini_RPAB.jsonl     # 1-intent subset (smoke test)
    └── step_<N>_result/        # Created by the pipeline
```

`workdir/` **must** be the current working directory whenever a `step_*`
script is invoked — every script asserts `cwd.name == "workdir"` and
resolves all input/output paths relative to it.

## 3. Installing dependencies

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the locked Python environment
cd proj-jailbreak-sandbox
uv sync

# 3. (Required by the attack library) install easyjailbreak and a few
#    helpers that are not on PyPI as direct deps of pyproject.toml.
uv pip install easyjailbreak llama-recipes datasets

# 4. (Optional) provide an OpenAI key for GPT-judge / GPTFuzzer / PAIR.
echo "OPENAI_API_KEY=sk-..." > .env
```

A Hugging Face token with access to gated models
(`meta-llama/*`, `meta-llama/Meta-Llama-Guard-2-8B`,
`meta-llama/Prompt-Guard-86M`, etc.) must also be available, e.g.:

```bash
huggingface-cli login
```

## 4. Models used

| Role | Models |
|------|--------|
| Target LLMs | Meta-Llama-3-70B-Instruct-AWQ, Qwen1.5-72B-Chat-AWQ, Phi-3-medium-128k-instruct |
| Draft SLMs | opt-125m-AWQ, SmolLM-135M, SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B, Llama-3.2-1B |
| Guards | LlamaGuardPrompt, LlamaGuardResponse, LlamaGuardPromptResponse, PromptGuard, PerplexityGuardPrompt, GradSafeGuardPrompt, AllTrueGuard |
| Attackers | PAIR, Cipher, DeepInception, GPTFuzzer, ICA |

The mapping from short names used on the CLI to Hugging Face model IDs
lives in [src/core/utils.py](src/core/utils.py).

## 5. Pipeline overview

The artifact is structured as a sequence of independent SLURM array jobs.
Each step reads pickles produced by the previous step and writes its own
`step_<N>_result/...` tree inside `workdir/`.

### Malicious-prompt pipeline (RPAB)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | [src/core/step_1_jailbreak_generation.py](src/core/step_1_jailbreak_generation.py) | Generate jailbreak prompts against each target model using each attack method |
| 2 | [src/core/step_2_target_model_inference.py](src/core/step_2_target_model_inference.py) | Run target LLM on the generated prompts |
| 3 | [src/core/step_3_draft_model_inference.py](src/core/step_3_draft_model_inference.py) | Run each draft SLM on the same prompts (`--draft_number` samples per prompt) |
| 4 | [src/core/step_4_target_response_ground_truth_guard_evaluation.py](src/core/step_4_target_response_ground_truth_guard_evaluation.py) | GPT-judge ground-truth labels for target responses |
| 5 | [src/core/step_5_target_response_guard_evaluation.py](src/core/step_5_target_response_guard_evaluation.py) | Baseline guards (LlamaGuard{Prompt,Response,PromptResponse}, PromptGuard, PerplexityGuardPrompt, GradSafeGuardPrompt, AllTrueGuard) on target responses |
| 6 | [src/core/step_6_draft_response_guard_evaluation.py](src/core/step_6_draft_response_guard_evaluation.py) | LlamaGuardResponse on draft responses (the **Sandbox** post-draft check) |
| 7 | [src/core/step_7_metric_calculation.py](src/core/step_7_metric_calculation.py) | ASR / DSR / latency tables across guard / draft / threshold settings |
| 8 | [src/core/step_8_malicious_analysis.ipynb](src/core/step_8_malicious_analysis.ipynb) | Plots and tables for the malicious-prompt experiments |

### Benign-prompt pipeline (Just-Eval)

| Step | Script | Purpose |
|------|--------|---------|
| 9 | [src/core/step_9_benign_target_model_inference.py](src/core/step_9_benign_target_model_inference.py) | Target LLM on benign Just-Eval prompts |
| 10 | [src/core/step_10_benign_draft_model_inference.py](src/core/step_10_benign_draft_model_inference.py) | Draft SLM on benign prompts |
| 11 | [src/core/step_11_benign_target_response_guard_evaluation.py](src/core/step_11_benign_target_response_guard_evaluation.py) | Baseline guards on benign target responses |
| 12 | [src/core/step_12_benign_draft_response_guard_evaluation.py](src/core/step_12_benign_draft_response_guard_evaluation.py) | LlamaGuardResponse on benign draft responses |
| 13 | [src/core/step_13_benign_analysis.py](src/core/step_13_benign_analysis.py) | Practicality / latency on benign prompts |

### Auxiliary

- [src/core/step_0_awq.py](src/core/step_0_awq.py) — AWQ-quantize draft
  models (`opt-125m`) used as small drop-in baselines.
- [src/core/step_14_cost_estimation.py](src/core/step_14_cost_estimation.py)
  — Token / dollar cost estimates.
- [src/core/step_15_transferability_matrix.ipynb](src/core/step_15_transferability_matrix.ipynb),
  [src/core/step_16_distribution.ipynb](src/core/step_16_distribution.ipynb) —
  Transferability and response-distribution analyses.

## 6. Running the pipeline

### 6.1 With SLURM (full reproduction)

For each step, [src/slurm/](src/slurm/) provides

- `<step>.tpl` — SLURM `sbatch` template,
- `<step>.yaml` — the experimental matrix (which target models, draft
  models, guards, attackers, …) to expand.

A small Jinja-based submission helper, [workdir/submit.py](workdir/submit.py),
expands the Cartesian product of one task's matrix and submits one SLURM
array job per combination. Run it from inside `workdir/`:

```bash
cd workdir
uv run python submit.py --task 1jg     # step 1: jailbreak generation
uv run python submit.py --task 2tmi    # step 2: target inference
uv run python submit.py --task 3dmi    # step 3: draft inference
uv run python submit.py --task 4trgtge # step 4: GPT-judge ground truth
uv run python submit.py --task 5trge   # step 5: target-response guards
uv run python submit.py --task 6drge   # step 6: draft-response guard
# Step 7 is a single local job (no .yaml/.tpl):
uv run ../src/core/step_7_metric_calculation.py
```

The benign pipeline mirrors this with tasks `9btmi`, `10bdmi`, `11btrge`,
`12bdrge`, followed by `step_13_benign_analysis.py`. Each step depends on
the pickles produced by the previous one, so wait for one matrix to
finish (`squeue -u $USER`) before submitting the next.

### 6.2 Without SLURM (manual)

Every script accepts the same arguments that the SLURM templates pass.
The required positional arguments are described in each script's
`argparse` block. The minimal end-to-end smoke test is:

```bash
cd workdir
uv run ../src/core/step_1_jailbreak_generation.py \
    --slurm_unit_index 0 \
    --target_model_name Phi-3-medium-128k-instruct \
    --generation_name DeepInception \
    --dataset_name mini_RPAB
```

`mini_RPAB` is a single-intent subset of RPAB included for fast
iteration.

### 6.3 Convenience runner

For artifact reviewers, a top-level driver is provided at
[run_artifact_evaluation.sh](run_artifact_evaluation.sh):

```bash
bash run_artifact_evaluation.sh check            # env + deps + GPU only
bash run_artifact_evaluation.sh smoke            # local 1-intent end-to-end
bash run_artifact_evaluation.sh slurm 1jg        # submit one SLURM matrix
bash run_artifact_evaluation.sh slurm-malicious  # full RPAB pipeline (1jg..6drge)
bash run_artifact_evaluation.sh slurm-benign     # full benign pipeline
```

`smoke` runs entirely without SLURM on a single GPU; the `slurm-*` modes
chain `submit.py` calls in dependency order and pause between steps for
the user to confirm completion (since `submit.py` does not chain SLURM
job dependencies). See `bash run_artifact_evaluation.sh --help` for
details.

## 7. Reproducing the paper claims

| Paper claim | Steps to run | Notebook / script that produces the figure |
|-------------|--------------|---------------------------------------------|
| Transferability matrix (Sec. 5) | 1 → 2 → 3 → 4 → 6 | step_15_transferability_matrix.ipynb |
| ASR / DSR vs. baseline guards (Tab. 2–3) | 1 → 2 → 3 → 4 → 5 → 6 → 7 | step_8_malicious_analysis.ipynb |
| Latency reduction vs. post-model guards | 1 → 2 → 3 → 5 → 6 → 7 | step_8_malicious_analysis.ipynb |
| Practicality / latency on benign prompts | 9 → 10 → 11 → 12 → 13 | step_13_benign_analysis.py / .ipynb |
| Threshold sensitivity (Appendix) | 7 (the outer loop sweeps thresholds 0.0–0.9) | step_8_malicious_analysis.ipynb |

## 8. Datasets

- **RPAB** (`workdir/dataset/RPAB.jsonl`): 50 malicious intents across 7
  categories (Cyberbullying, Fraud, Hacking, Illegal Drug Use, Theft,
  Violence, Misinformation). The full enumeration is given in
  Appendix Table 1 of the paper.
- **mini_RPAB** (`workdir/dataset/mini_RPAB.jsonl`): 1 intent, used for
  the kick-the-tires smoke test.
- **Just-Eval** (benign): downloaded separately by step 9; the original
  SLURM job loaded it from `/depot/anonymous/data/.../just-eval/`.
  Edit the path in `step_9_benign_target_model_inference.py` before
  running outside the original cluster.

## 9. License

Released under the LICENSE in this repository for academic use.

## 10. Citation

If you use this artifact, please cite the paper:

```bibtex
@inproceedings{cai2026sandbox,
  title     = {Exploring and Developing a Pre-Model Safeguard with Draft Models},
  author    = {Cai, Hongyu and Arunasalam, Arjun and Liang, Yiming and
               Bianchi, Antonio and Celik, Z. Berkay},
  booktitle = {Proceedings of the ACM Conference},
  year      = {2026},
}
```

## 11. Contact

Hongyu Cai &lt;hongyu@purdue.edu&gt; — for questions related to the
artifact or the paper.
