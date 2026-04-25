#!/usr/bin/env bash
#
# run_artifact_evaluation.sh
#
# Driver for the artifact evaluation of "Exploring and Developing a
# Pre-Model Safeguard with Draft Models". Performs environment checks
# and runs the multi-step pipeline so an AE reviewer can validate the
# toolchain on a single machine, or fan the full experiment matrix out
# across a SLURM cluster.
#
# Modes:
#   check                 environment + dependency checks only.
#
#   smoke      (default)  one intent (mini_RPAB) + Phi-3 + SmolLM-135M +
#                         DeepInception + LlamaGuard{Prompt,Response,
#                         PromptResponse}. Runs locally without SLURM,
#                         no sbatch required. Finishes in a few hours
#                         on a single A100-80GB.
#
#   slurm <task>          submit a single SLURM matrix described by
#                         src/slurm/<task>.{yaml,tpl} via workdir/submit.py.
#                         <task> is one of: 1jg | 2tmi | 3dmi | 4trgtge |
#                         5trge | 6drge | 9btmi | 10bdmi | 11btrge |
#                         12bdrge.
#
#   slurm-malicious       submit the full malicious pipeline (1jg ... 6drge)
#                         to SLURM, in dependency order. Each step is
#                         submitted only after the user confirms the
#                         previous one finished, since submit.py does
#                         not chain SLURM job dependencies.
#                         Run step 7 locally afterwards.
#
#   slurm-benign          submit the full benign pipeline (9btmi ... 12bdrge)
#                         to SLURM, same caveat about chaining as above.
#                         Run step 13 locally afterwards.
#
# Usage:  bash run_artifact_evaluation.sh [check|smoke|slurm <task>|
#                                          slurm-malicious|slurm-benign]
#

set -euo pipefail

MODE="${1:-smoke}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${REPO_ROOT}/workdir"
CORE="${REPO_ROOT}/src/core"
SUBMIT="${WORKDIR}/submit.py"

log()  { printf '\033[1;34m[ae]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ae]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ae]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

case "${MODE}" in
    -h|--help|help) usage ;;
esac

############################################
# 0. Environment checks
############################################

log "Checking environment..."

command -v uv >/dev/null 2>&1 \
    || die "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"

[[ -d "${WORKDIR}" ]] \
    || die "workdir/ not found at ${WORKDIR}"
[[ -f "${SUBMIT}" ]] \
    || die "workdir/submit.py is missing"

[[ -f "${WORKDIR}/dataset/RPAB.jsonl" ]] \
    || die "RPAB dataset missing at workdir/dataset/RPAB.jsonl"
[[ -f "${WORKDIR}/dataset/mini_RPAB.jsonl" ]] \
    || die "mini_RPAB dataset missing at workdir/dataset/mini_RPAB.jsonl"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found - this artifact requires a CUDA GPU."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
        | sed 's/^/[ae]   GPU: /'
fi

log "Syncing Python environment with uv (this may take a while on first run)..."
( cd "${REPO_ROOT}" && uv sync )

# Extra deps not pinned in pyproject.toml:
#   easyjailbreak / llama-recipes / datasets are required by step_1,
#   step_4 and the GradSafe guard; jinja2 + pyyaml are required by
#   workdir/submit.py.
log "Ensuring extra dependencies (easyjailbreak, llama-recipes, datasets, jinja2, pyyaml)..."
( cd "${REPO_ROOT}" \
  && uv pip install --quiet easyjailbreak llama-recipes datasets jinja2 pyyaml )

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    warn ".env not found. PAIR / GPTFuzzer attackers and the GPT-judge step"
    warn "(step 4) will fail without OPENAI_API_KEY. Create .env with:"
    warn "    echo 'OPENAI_API_KEY=sk-...' > .env"
fi

if [[ ! -f "${HOME}/.cache/huggingface/token" ]]; then
    warn "Hugging Face credentials not detected. Run 'huggingface-cli login'"
    warn "to access gated models (Meta-Llama, Llama-Guard, Prompt-Guard)."
fi

log "Environment looks OK."

if [[ "${MODE}" == "check" ]]; then
    exit 0
fi

############################################
# Helpers
############################################

# Run a step_*.py script locally (every script must run from workdir/).
run_step() {
    local script="$1"; shift
    log "  $ uv run ${script} $*"
    ( cd "${WORKDIR}" && uv run "${CORE}/${script}" "$@" )
}

require_sbatch() {
    command -v sbatch >/dev/null 2>&1 \
        || die "sbatch not found. Use 'smoke' mode for a non-SLURM run."
}

# Submit one matrix described by src/slurm/<task>.{yaml,tpl} via submit.py.
submit_task() {
    local task="$1"
    require_sbatch
    log "Submitting SLURM matrix: ${task}"
    ( cd "${WORKDIR}" && uv run python submit.py --task "${task}" )
}

# Pause until the user confirms the previously-submitted SLURM array has
# finished. This is needed because submit.py does not chain dependencies
# between steps -- each step's pickles must exist before the next one
# can be submitted.
wait_for_user() {
    local prev="$1" next="$2"
    cat <<EOF

  ===> ${prev} has been submitted to SLURM.
       Wait for all of its array tasks to finish (squeue -u \$USER), then
       press [Enter] to submit ${next}, or Ctrl-C to abort.

EOF
    read -r _
}

############################################
# 1. Local smoke test --- malicious pipeline on mini_RPAB (1 intent)
############################################

smoke_pipeline() {
    local DATASET="mini_RPAB"
    local TARGET="Phi-3-medium-128k-instruct"
    local DRAFT="SmolLM-135M"
    local ATTACK="DeepInception"
    local DRAFT_NUM=20
    local INTENT_IDX=0

    log "=== SMOKE: ${ATTACK} | ${TARGET} | ${DRAFT} (n=${DRAFT_NUM}) | ${DATASET} ==="

    log "Step 1: jailbreak generation"
    run_step step_1_jailbreak_generation.py \
        --slurm_unit_index "${INTENT_IDX}" \
        --target_model_name "${TARGET}" \
        --generation_name   "${ATTACK}" \
        --dataset_name      "${DATASET}"

    log "Step 2: target model inference"
    run_step step_2_target_model_inference.py \
        --slurm_unit_index "${INTENT_IDX}" \
        --target_model_name "${TARGET}" \
        --generation_name   "${ATTACK}" \
        --dataset_name      "${DATASET}"

    log "Step 3: draft model inference"
    run_step step_3_draft_model_inference.py \
        --slurm_unit_index "${INTENT_IDX}" \
        --target_model_name "${TARGET}" \
        --draft_model_name  "${DRAFT}" \
        --generation_name   "${ATTACK}" \
        --dataset_name      "${DATASET}" \
        --draft_number      "${DRAFT_NUM}"

    if [[ -f "${REPO_ROOT}/.env" ]]; then
        log "Step 4: GPT-judge ground truth"
        run_step step_4_target_response_ground_truth_guard_evaluation.py \
            --slurm_unit_index "${INTENT_IDX}" \
            --target_model_name "${TARGET}" \
            --generation_name   "${ATTACK}" \
            --dataset_name      "${DATASET}"
    else
        warn "Step 4 skipped (no OPENAI_API_KEY)."
    fi

    for guard in LlamaGuardPrompt LlamaGuardResponse LlamaGuardPromptResponse; do
        log "Step 5: target-response guard (${guard})"
        run_step step_5_target_response_guard_evaluation.py \
            --slurm_unit_index "${INTENT_IDX}" \
            --target_model_name "${TARGET}" \
            --generation_name   "${ATTACK}" \
            --dataset_name      "${DATASET}" \
            --guard_name        "${guard}"
    done

    log "Step 6: draft-response guard (LlamaGuardResponse)"
    run_step step_6_draft_response_guard_evaluation.py \
        --slurm_unit_index "${INTENT_IDX}" \
        --target_model_name "${TARGET}" \
        --draft_model_name  "${DRAFT}" \
        --generation_name   "${ATTACK}" \
        --dataset_name      "${DATASET}" \
        --guard_name        "LlamaGuardResponse"

    log "Smoke pipeline finished."
    log "Inspect intermediate pickles under workdir/step_<N>_result/${DATASET}/..."
    log "Step 7 (metrics) and step 8 (notebook) require the full RPAB matrix;"
    log "use 'slurm-malicious' on a SLURM cluster for full reproduction."
}

############################################
# 2. SLURM: full malicious-prompt reproduction (RPAB)
############################################

slurm_malicious() {
    log "=== FULL MALICIOUS pipeline via SLURM ==="
    log "Six SLURM matrices will be submitted in order. submit.py does not"
    log "chain SLURM dependencies, so the script will pause between steps"
    log "for you to confirm completion (squeue -u \$USER)."

    local STEPS=( 1jg 2tmi 3dmi 4trgtge 5trge 6drge )
    local n=${#STEPS[@]}

    for ((i=0; i<n; i++)); do
        submit_task "${STEPS[$i]}"
        if (( i < n - 1 )); then
            wait_for_user "${STEPS[$i]}" "${STEPS[$((i+1))]}"
        fi
    done

    cat <<EOF

  All SLURM matrices submitted. After 6drge finishes, run step 7
  locally to compute the ASR / DSR / latency tables:

      cd workdir && uv run ../src/core/step_7_metric_calculation.py

  Then open src/core/step_8_malicious_analysis.ipynb to regenerate
  Tables 2-3 and Figure 5 of the paper.

EOF
}

############################################
# 3. SLURM: benign-prompt pipeline (Just-Eval)
############################################

slurm_benign() {
    log "=== FULL BENIGN pipeline via SLURM ==="
    warn "step_9 expects Just-Eval pickles at the path baked into"
    warn "src/core/step_9_benign_target_model_inference.py:55 -- update that"
    warn "path to point to your local Just-Eval copy before submitting."

    local STEPS=( 9btmi 10bdmi 11btrge 12bdrge )
    local n=${#STEPS[@]}

    for ((i=0; i<n; i++)); do
        submit_task "${STEPS[$i]}"
        if (( i < n - 1 )); then
            wait_for_user "${STEPS[$i]}" "${STEPS[$((i+1))]}"
        fi
    done

    cat <<EOF

  All benign SLURM matrices submitted. After 12bdrge finishes, run
  step 13 locally to produce the practicality / latency analysis:

      cd workdir && uv run ../src/core/step_13_benign_analysis.py

  The matching notebook (step_13_benign_analysis.ipynb) regenerates
  the figures of Sec. 6.4.

EOF
}

############################################
# Dispatch
############################################

case "${MODE}" in
    smoke)
        smoke_pipeline ;;
    slurm)
        TASK="${2:-}"
        [[ -n "${TASK}" ]] \
            || die "Usage: $0 slurm <task>  (e.g. 1jg, 2tmi, ...)"
        submit_task "${TASK}"
        ;;
    slurm-malicious)
        slurm_malicious ;;
    slurm-benign)
        slurm_benign ;;
    *)
        die "unknown mode: ${MODE}  (try: check | smoke | slurm <task> | slurm-malicious | slurm-benign)"
        ;;
esac

log "Done."
