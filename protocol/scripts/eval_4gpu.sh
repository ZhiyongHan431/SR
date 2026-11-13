#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE=""
NEGATIVE_FRACTION="0.02"
RUN_TAG=""
CHECKPOINT=""
CFG=""
PHASE_ARGS=()
PHASE_NAME="generate_and_metrics"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --negative-fraction)
      NEGATIVE_FRACTION="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --cfg)
      CFG="$2"
      shift 2
      ;;
    --generate-only)
      PHASE_ARGS=(--generate_only)
      PHASE_NAME="generate_only"
      shift
      ;;
    --metrics-only)
      PHASE_ARGS=(--metrics_only)
      PHASE_NAME="metrics_only"
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/eval_4gpu.sh --mode no_cfg|synthetic_cfg [options]"
      echo "  --negative-fraction 0.01..0.05"
      echo "  --run-tag NAME"
      echo "  --checkpoint PATH"
      echo "  --cfg VALUE"
      echo "  --generate-only | --metrics-only"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "no_cfg" && "${MODE}" != "synthetic_cfg" ]]; then
  echo "--mode must be no_cfg or synthetic_cfg" >&2
  exit 2
fi
if [[ "${NEGATIVE_FRACTION}" =~ ^0\.0([1-5])0*$ ]]; then
  NEGATIVE_PERCENT="${BASH_REMATCH[1]}"
  NEGATIVE_FRACTION="0.0${NEGATIVE_PERCENT}"
else
  echo "--negative-fraction must be 0.01 through 0.05" >&2
  exit 2
fi
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "--run-tag may contain only letters, digits, underscore, and hyphen" >&2
  exit 2
fi

if [[ "${MODE}" == "no_cfg" ]]; then
  EXP_NAME="VARSR_LSDIR_FFHQ10K_no_cfg"
  REPORT_NEGATIVE_FRACTION="0.0"
else
  FRACTION_TAG="${NEGATIVE_FRACTION//./p}"
  EXP_NAME="VARSR_LSDIR_FFHQ10K_synthetic_cfg_f${FRACTION_TAG}"
  REPORT_NEGATIVE_FRACTION="${NEGATIVE_FRACTION}"
fi
if [[ -n "${RUN_TAG}" ]]; then
  EXP_NAME="${EXP_NAME}_${RUN_TAG}"
fi
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="results/${EXP_NAME}/ar-ckpt-best.pth"
fi

NUM_GPUS="${NUM_GPUS:-4}"
if [[ "${NUM_GPUS}" -ne 4 ]]; then
  echo "This launcher requires NUM_GPUS=4 for the requested protocol." >&2
  exit 2
fi
if [[ -n "${CFG}" && ! "${CFG}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--cfg must be a non-negative number" >&2
  exit 2
fi

CFG_ARGS=()
if [[ -n "${CFG}" ]]; then
  CFG_ARGS=(--cfg "${CFG}")
  CFG_EFFECTIVE="${CFG}"
elif [[ "${MODE}" == "no_cfg" ]]; then
  CFG_EFFECTIVE="0"
else
  CFG_EFFECTIVE="6"
fi
CFG_TAG="${CFG_EFFECTIVE//./p}"
OUTPUT_ROOT="outputs/alignvar_eval/${EXP_NAME}_cfg${CFG_TAG}"
METRICS_PATH="${OUTPUT_ROOT}/${MODE}/metrics.json"

RUN_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="reports/${EXP_NAME}"
LOG_DIR="${REPORT_DIR}/logs"
EVAL_LOG="${LOG_DIR}/eval_${PHASE_NAME}_cfg${CFG_TAG}_${RUN_UTC}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${EVAL_LOG}") 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

report_event() {
  python scripts/experiment_report.py event \
    --experiment "${EXP_NAME}" \
    --mode "${MODE}" \
    --negative-fraction "${REPORT_NEGATIVE_FRACTION}" \
    --phase evaluation \
    --event "$1" \
    --status "$2" \
    --log-path "${EVAL_LOG}" \
    --checkpoint "${CHECKPOINT}" \
    --metrics-path "${METRICS_PATH}" \
    --exit-code "$3" \
    --command "${EVAL_COMMAND:-}"
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "[$(timestamp)] EVALUATION LAUNCH"
echo "experiment=${EXP_NAME}"
echo "mode=${MODE}; cfg=${CFG_EFFECTIVE}; phase=${PHASE_NAME}"
echo "checkpoint=${CHECKPOINT}"
echo "output_root=${OUTPUT_ROOT}"
echo "log=${EVAL_LOG}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  EVAL_COMMAND="checkpoint validation"
  report_event evaluation_refused failed 1
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

EVAL_CMD=(
  torchrun
  --standalone
  --nproc-per-node="${NUM_GPUS}"
  scripts/evaluate_alignvar.py
  --checkpoint "${CHECKPOINT}"
  --mode "${MODE}"
  --vae checkpoints/VQVAE.pth
  --output_root "${OUTPUT_ROOT}"
  --crop_border 0
  --top_k 1
  --top_p 0.75
  --seed 0
  --metric_workers 8
  --metric_batch_size 32
  "${CFG_ARGS[@]}"
  "${PHASE_ARGS[@]}"
)
printf -v EVAL_COMMAND '%q ' "${EVAL_CMD[@]}"

EVAL_ACTIVE=1
interrupt_evaluation() {
  local exit_code="$1"
  local signal_name="$2"
  if [[ "${EVAL_ACTIVE}" -eq 1 ]]; then
    echo "[$(timestamp)] EVALUATION INTERRUPTED by ${signal_name}; exit=${exit_code}"
    report_event evaluation_interrupted interrupted "${exit_code}" || true
    EVAL_ACTIVE=0
  fi
  exit "${exit_code}"
}
trap 'interrupt_evaluation 130 SIGINT' INT
trap 'interrupt_evaluation 143 SIGTERM' TERM
trap 'interrupt_evaluation 129 SIGHUP' HUP

report_event evaluation_started running -
echo "[$(timestamp)] verify bundled metric cache"
set +e
python scripts/install_metric_cache.py --verify_existing
CACHE_RC=$?
set -e
if [[ "${CACHE_RC}" -ne 0 ]]; then
  EVAL_ACTIVE=0
  report_event metric_cache_failed failed "${CACHE_RC}"
  echo "[$(timestamp)] metric cache verification failed; exit=${CACHE_RC}" >&2
  exit "${CACHE_RC}"
fi

echo "[$(timestamp)] torchrun evaluation started; output is mirrored to ${EVAL_LOG}"
echo "Existing valid predictions with the same generation signature will be skipped."
set +e
"${EVAL_CMD[@]}"
EVAL_RC=$?
set -e
EVAL_ACTIVE=0
trap - INT TERM HUP

if [[ "${EVAL_RC}" -eq 0 ]]; then
  report_event "evaluation_${PHASE_NAME}_completed" completed 0
  echo "[$(timestamp)] OK: evaluation phase complete"
  if [[ "${PHASE_NAME}" != "generate_only" ]]; then
    echo "metrics: ${METRICS_PATH}"
  fi
  echo "report: ${REPORT_DIR}/REPORT.md"
else
  if [[ "${EVAL_RC}" -eq 130 || "${EVAL_RC}" -eq 143 || "${EVAL_RC}" -eq 129 ]]; then
    EVAL_STATUS="interrupted"
    EVAL_EVENT="evaluation_interrupted"
  else
    EVAL_STATUS="failed"
    EVAL_EVENT="evaluation_failed"
  fi
  report_event "${EVAL_EVENT}" "${EVAL_STATUS}" "${EVAL_RC}"
  echo "[$(timestamp)] EVALUATION ${EVAL_STATUS}; exit=${EVAL_RC}; rerun the same command to continue generation" >&2
  exit "${EVAL_RC}"
fi
