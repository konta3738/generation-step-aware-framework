#!/usr/bin/env bash
# main_prod_env.sh — portable launcher for compute_responses/expertise pipelines
# - Works on Colab (no conda/modules) and HPC (with conda/modules)
# - Accepts: either ONE quoted string or normal space-separated args
# - Provides safer memory defaults + clearer logging
set -x #debug for oom problem
set -Eeuo pipefail

### ---------- Logging helpers ----------
log()   { printf "[INFO] %s\n" "$*" >&2; }
warn()  { printf "[WARN] %s\n" "$*" >&2; }
error() { printf "[ERROR] %s\n" "$*" >&2; }
die()   { error "$*"; exit 1; }

### ---------- Usage ----------
usage() {
  cat >&2 <<'USAGE'
Usage:
  bash ./main_prod_env.sh \
    MODEL_NAME PHASE DATAPATH LANGUAGE NUM_UNITS FORCE EXPERT_FILE \
    TASK PROMPT_ID SRC_LANG TGT_LANG SAMPLING_RATE

Or (legacy) as one quoted string:
  bash ./main_prod_env.sh "MODEL_NAME PHASE DATAPATH LANGUAGE NUM_UNITS FORCE EXPERT_FILE TASK PROMPT_ID SRC_LANG TGT_LANG SAMPLING_RATE"

Examples:
  bash ./main_prod_env.sh \
    seamless-m4t-v2-large compute_responses \
    Speech ja_speech_VC 2000 on_p50 expertise_limited_2000_both \
    s2t_translation "" jpn eng 16000

Minimal (let defaults fill in):
  bash ./main_prod_env.sh "seamless-m4t-v2-large compute_responses Speech ja_speech_VC 2000 on_p50 expertise_limited_2000_both s2t_translation '' jpn eng 16000"
USAGE
}

### ---------- Parse args (1 big string OR normal) ----------
if [[ $# -eq 0 ]]; then
  usage; die "No arguments provided."
fi

if [[ $# -eq 1 ]]; then
  # Caller passed one giant string; split safely
  read -r -a args <<< "$1"
else
  # Caller used normal multi-arg calling convention
  args=("$@")
fi

# Positional arguments (use defaults to avoid -u errors)
model_name="${args[0]:-}"
phase="${args[1]:-}"
datapath="${args[2]:-}"             # e.g., "Speech"
language="${args[3]:-}"             # e.g., "ja_speech_VC"
num_units="${args[4]:-}"
force_value="${args[5]:-}"
expert_file="${args[6]:-}"
task_in="${args[7]:-}"              # e.g., ASR | s2t_translation | t2t_translation (optional)
prompt_format_id_for_translation="${args[8]:-}"  # used only by generate_activated_condition
src_lang="${args[9]:-}"
tgt_lang="${args[10]:-}"
sampling_rate="${args[11]:-}"
expertise_path_override="${args[12]:-}"
results_path_override="${args[13]:-}"

# Basic validation
[[ -z "${model_name}" ]] && die "MODEL_NAME is required."
[[ -z "${phase}"      ]] && die "PHASE is required."
[[ -z "${datapath}"   ]] && die "DATAPATH is required."
#[[ -z "${language}"   ]] && die "LANGUAGE is required."
if [[ -z "${language}" ]]; then
  # allow empty LANGUAGE only for limit_expertise (process all languages)
  if [[ "${phase}" != "limit_expertise" && "${phase}" != "compute_all" ]]; then
    die "LANGUAGE is required."
  fi
fi

### ---------- Print parsed config ----------
printf "%s\n" "${args[*]}"
log "Parsed arguments:"
log "  model_name=${model_name}"
log "  phase=${phase}"
log "  datapath=${datapath}"
log "  language=${language}"
log "  num_units=${num_units}"
log "  force_value=${force_value}"
log "  expert_file=${expert_file}"
log "  task_in=${task_in}"
log "  prompt_format_id_for_translation=${prompt_format_id_for_translation}"
log "  src_lang=${src_lang}"
log "  tgt_lang=${tgt_lang}"
log "  sampling_rate=${sampling_rate}"

### ---------- Safe defaults & memory-friendly env ----------
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
# Tame CUDA allocator fragmentation; safe even on CPU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

# Allow user to append extra flags to the Python call via env var
# Example: export EXTRA_COMPUTE_FLAGS="--load-in-8bit" (only if your script supports it)
EXTRA_COMPUTE_FLAGS="${EXTRA_COMPUTE_FLAGS:-}"

### ---------- Python command selection ----------
# If caller sets PYTHON (e.g., /venv/main/bin/python3), use it.
# Otherwise fall back to python/python3 on PATH.
PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY="python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    die "python/python3 not found on PATH (and PYTHON not set)."
  fi
fi
log "Using PY=${PY}"

#below is for vast.ai above is for google colab
### ---------- Optional: conda + environment modules ----------
if [[ -z "${PYTHON:-}" ]] && command -v conda >/dev/null 2>&1; then  # Only source ~/.bashrc if it exists and ignore PS1 errors
  if [ -f ~/.bashrc ]; then
    # Temporarily disable -u so sourcing doesn't break on unset vars like PS1
    set +u
    # shellcheck disable=SC1090
    source ~/.bashrc || true
    set -u
  fi
  conda activate lang_neuron 2>/dev/null || warn "Conda env 'lang_neuron' not found; continuing."
else
  warn "conda not found; skipping activation (expected on Colab or micromamba)."
fi

if command -v module >/dev/null 2>&1; then
  module load gcc/8.3.1 gcc/8.5.0 cuda/11.7/11.7.1 cudnn/8.8/8.8.1 2>/dev/null || \
    warn "module load failed or not needed; continuing."
else
  warn "'module' command not found; skipping (expected on Colab)."
fi

### ---------- Paths (EDIT to match your repo layout) ----------
model_path="set_appropriate_path_1/"
base_path="set_appropriate_path_2/"
expert_base_path="expertise_calculation/"
#model_dir="${model_name2##*/}"

#override the repo, for exp2 (generated_activate_conditon)
condition_base_path="${CONDITION_BASE_PATH:-}"

### ---------- Model family detection ----------
shopt -s nocasematch
is_seamless=0
prompt=""

if [[ "${model_name}" == *xglm* ]]; then
  model_name2="facebook/${model_name}"
  prompt=""
elif [[ "${model_name}" == *bloom* ]]; then
  model_name2="bigscience/${model_name}"
  prompt="</s>"
elif [[ "${model_name}" == *Llama-2* ]]; then
  model_name2="${model_path}llama2-HF/${model_name}"
  prompt=""
elif [[ "${model_name}" == *seamless* && "${model_name}" == *m4t* ]]; then
  model_name2="facebook/${model_name}"
  prompt=""
  is_seamless=1
else
  die "Unknown model family: ${model_name}"
fi
shopt -u nocasematch

### ---------- Task normalization & defaults ----------
task_norm="${task_in:-}"
if [[ -z "${task_norm}" ]]; then
  if [[ ${is_seamless} -eq 1 ]]; then
    task_norm="ASR"               # default for Seamless if unspecified
  else
    task_norm="t2t_translation"   # default for text LMs
  fi
fi

# Default sampling rate
if [[ -z "${sampling_rate}" ]]; then
  sampling_rate="16000"
fi

# If ASR and tgt_lang empty -> tgt=src
if [[ "${task_norm,,}" == "asr" && -n "${src_lang}" && -z "${tgt_lang}" ]]; then
  tgt_lang="${src_lang}"
fi

# Task label for directory
task_slug="$(echo "${task_norm}" | tr '[:upper:]' '[:lower:]')"

case "$task_norm" in
  en2x_s2t_translation) task_norm="s2t_translation" ;;
  en2x_t2t_translation) task_norm="t2t_translation" ;;
esac

# Compose responses base (task-scoped)
# Result: set_appropriate_path_2/Speech/<task_slug>/facebook/seamless-m4t-v2-large/sense/...
responses_base="${base_path}${datapath}/${task_slug}"
expert_responses_base="${expert_base_path}${datapath}/${task_slug}"

responses_root="${responses_base}"

### ---------- Phase: compute_responses ----------
if [[ "${phase}" == "compute_responses" || "${phase}" == "compute_all" ]]; then
  log "Running compute_responses with task=${task_norm}, src=${src_lang}, tgt=${tgt_lang}, sr=${sampling_rate}"

  cmd=( python3 scripts/compute_responses.py
    --model-name-or-path "${model_name2}"
    --data-path "assets/${datapath}"
    --responses-path "${responses_base}"
    --concepts "sense/${language}"
    --seq-len 256
    --inf-batch-size 1
    --task "${task_norm}"
    --sampling-rate "${sampling_rate}"
  )

  # Pass langs only when provided
  if [[ -n "${src_lang}" ]]; then
    cmd+=( --src-lang "${src_lang}" )
  fi
  if [[ -n "${tgt_lang}" ]]; then
    cmd+=( --tgt-lang "${tgt_lang}" )
  fi

  # Optional extra flags via env
  if [[ -n "${EXTRA_COMPUTE_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    extra=( ${EXTRA_COMPUTE_FLAGS} )
    cmd+=( "${extra[@]}" )
  fi

  # Launch
  "${cmd[@]}"
fi

### ---------- Phase: compute_expertise ----------
if [[ "${phase}" == "compute_expertise" || "${phase}" == "compute_all" ]]; then
  python3 scripts/compute_expertise.py \
    --root-dir "${responses_base}" \
    --model-name "${model_name}" \
    --concepts "sense/${language}"
fi

### ---------- Phase: limit_expertise ----------
if [[ "${phase}" == "limit_expertise" || "${phase}" == "compute_all" ]]; then
  cmd=( python3 scripts/make_limited_expert_exe.py
    --model-name "${model_name}" \
    --num-units "${num_units}" \
    --root-dir "${base_path%/}" \
    --task "${task_slug}"
  )
  if [[ -n "${language}" ]]; then
    cmd+=( --language "${language}" )
  fi
  "${cmd[@]}"
fi
#--out-dir "${responses_base}/${model_name}/sense/${language}/expertise"

#--root-dir "./expertise_calculation"

### ---------- Phase: generate_activated ----------
if [[ "${phase}" == "generate_activated" ]]; then
  if [[ ${is_seamless} -eq 1 ]]; then
    log "generate_activated is not supported for SeamlessM4T. Skipping."
  else
    python3 scripts/generate_seq_lang.py \
      --model-name-or-path "${model_name2}" \
      --expertise "${responses_base}/${model_name}/sense/${language}/expertise/${expert_file}.csv" \
      --length 64 \
      --seed 1 101 \
      --metric ap \
      --forcing "${force_value}" \
      --num-units "${num_units}" \
      --eos \
      --top-n 1 \
      --results-file "${responses_base}/${model_name}/sense/${language}/expertise/created_sentence_${force_value}_${num_units}_${expert_file}.csv" \
      --temperature 0.8 \
      --prompt "${prompt}"
  fi
fi

# === expertise path ===
if [[ -n "${expertise_path_override}" ]]; then
  expertise_path="${expertise_path_override}"
else
  expertise_path="${responses_root}/${model_name}/sense/${language}/expertise/${expert_file}.csv"
fi

# === results path ===
if [[ -n "${results_path_override}" ]]; then
  results_path="${results_path_override}"
else
  results_path="${responses_root}/${model_name}/sense/${language}/expertise/created_sentence_${force_value}_${num_units}_${expert_file}_${task_norm}_condition_${prompt_format_id_for_translation}.csv"
fi

### ---------- Phase: generate_activated_condition ----------
if [[ "${phase}" == "generate_activated_condition" ]]; then

  # === 入力ファイルの選択 ===
  # Data input: keep using assets/${datapath} (e.g. assets/Speech)
  input_file="assets/${datapath}/sense/${language}.json"

  # --- Decide where expertise + outputs live ---
  # default: same as other phases (set_appropriate_path_2/${datapath}/${task_slug})
  responses_root="${responses_base}"

  # if CONDITION_BASE_PATH is set, override to e.g. set_appropriate_path_5/<task_slug>
  if [[ -n "${condition_base_path}" ]]; then
    responses_root="${condition_base_path%/}/${task_slug}"
  fi

  cmd=( python3 scripts/generate_seq_lang.py
    --model-name-or-path "${model_name2}"
    --expertise "${expertise_path}"
    --length 128
    --seed 1 101
    --metric ap
    --forcing "${force_value}"
    --num-units "${num_units}"
    --eos
    --top-n 1
    --results-file "${results_path}"
    --temperature 0.0
    --prompt "${input_file}"
    --prompt_format_id_for_translation "${prompt_format_id_for_translation}"
    --task "${task_norm}"
  )

  if [[ ${is_seamless} -eq 1 ]]; then
    [[ -n "${src_lang}" ]] && cmd+=( --src-lang "${src_lang}" )
    [[ -n "${tgt_lang}" ]] && cmd+=( --tgt-lang "${tgt_lang}" )
    cmd+=( --seamless )
  fi

  "${cmd[@]}"
fi
