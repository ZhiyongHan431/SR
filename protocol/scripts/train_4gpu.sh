#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

MODE=""
NEGATIVE_FRACTION="0.02"
RESUME_POLICY="auto"
RUN_TAG=""
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
    --resume)
      RESUME_POLICY="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/train_4gpu.sh --mode no_cfg|synthetic_cfg [options]"
      echo "  --negative-fraction 0.01..0.05"
      echo "  --resume auto|required|never   (default: auto)"
      echo "  --run-tag NAME                 append a safe unique experiment suffix"
      echo "Optional environment: NUM_GPUS=4 MICRO_BATCH=8 ACCUM_STEPS=1 WORKERS=8 SEED=0"
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
  echo "--negative-fraction must be 0.01, 0.02, 0.03, 0.04, or 0.05" >&2
  exit 2
fi
if [[ "${RESUME_POLICY}" != "auto" && "${RESUME_POLICY}" != "required" && "${RESUME_POLICY}" != "never" ]]; then
  echo "--resume must be auto, required, or never" >&2
  exit 2
fi
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "--run-tag may contain only letters, digits, underscore, and hyphen" >&2
  exit 2
fi

NUM_GPUS="${NUM_GPUS:-4}"
MICRO_BATCH="${MICRO_BATCH:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
WORKERS="${WORKERS:-8}"
EPOCHS=5

for value in "${NUM_GPUS}" "${MICRO_BATCH}" "${ACCUM_STEPS}" "${WORKERS}" "${EPOCHS}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "GPU/batch/worker/epoch values must be non-negative integers" >&2
    exit 2
  fi
done

FORWARD_GLOBAL_BATCH=$((NUM_GPUS * MICRO_BATCH))
EFFECTIVE_BATCH=$((FORWARD_GLOBAL_BATCH * ACCUM_STEPS))
if [[ "${EFFECTIVE_BATCH}" -ne 32 ]]; then
  echo "Official README effective global batch is 32; got ${NUM_GPUS}*${MICRO_BATCH}*${ACCUM_STEPS}=${EFFECTIVE_BATCH}" >&2
  exit 2
fi
if [[ "${NUM_GPUS}" -ne 4 ]]; then
  echo "This launcher is intended for the requested four-GPU machine (NUM_GPUS=4)." >&2
  exit 2
fi

case "${ACCUM_STEPS}" in
  1) BASE_LR="5e-5" ;;
  2) BASE_LR="2.5e-5" ;;
  4) BASE_LR="1.25e-5" ;;
  8) BASE_LR="6.25e-6" ;;
  *)
    echo "ACCUM_STEPS must be one of 1, 2, 4, 8" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "no_cfg" ]]; then
  EXP_NAME="VARSR_LSDIR_FFHQ10K_no_cfg"
  NEGATIVE_FRACTION_ARG="0.0"
else
  FRACTION_TAG="${NEGATIVE_FRACTION//./p}"
  EXP_NAME="VARSR_LSDIR_FFHQ10K_synthetic_cfg_f${FRACTION_TAG}"
  NEGATIVE_FRACTION_ARG="${NEGATIVE_FRACTION}"
fi
if [[ -n "${RUN_TAG}" ]]; then
  EXP_NAME="${EXP_NAME}_${RUN_TAG}"
fi

POSITIVE_TRAIN_IMAGES=94991
SYNTHETIC_NEGATIVE_IMAGES=0
if [[ "${MODE}" == "synthetic_cfg" ]]; then
  NEGATIVE_DENOMINATOR=$((100 - NEGATIVE_PERCENT))
  SYNTHETIC_NEGATIVE_IMAGES=$(((POSITIVE_TRAIN_IMAGES * NEGATIVE_PERCENT + NEGATIVE_DENOMINATOR - 1) / NEGATIVE_DENOMINATOR))
fi
TOTAL_TRAIN_IMAGES=$((POSITIVE_TRAIN_IMAGES + SYNTHETIC_NEGATIVE_IMAGES))
ITERS_PER_EPOCH=$(((TOTAL_TRAIN_IMAGES + FORWARD_GLOBAL_BATCH - 1) / FORWARD_GLOBAL_BATCH))
RUN_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="reports/${EXP_NAME}"
LOG_DIR="${REPORT_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/train_${RUN_UTC}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${TRAIN_LOG}") 2>&1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

report_event() {
  python scripts/experiment_report.py event \
    --experiment "${EXP_NAME}" \
    --mode "${MODE}" \
    --negative-fraction "${NEGATIVE_FRACTION_ARG}" \
    --phase training \
    --event "$1" \
    --status "$2" \
    --resume-policy "${RESUME_POLICY}" \
    --log-path "${TRAIN_LOG}" \
    --checkpoint "results/${EXP_NAME}/ar-ckpt-best.pth" \
    --exit-code "$3" \
    --command "${TRAIN_COMMAND:-}"
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "[$(timestamp)] TRAIN LAUNCH"
echo "mode=${MODE}"
echo "experiment=${EXP_NAME}"
echo "resume_policy=${RESUME_POLICY}"
echo "log=${TRAIN_LOG}"
echo "forward_global_batch=${FORWARD_GLOBAL_BATCH}; gradient_accumulation=${ACCUM_STEPS}; effective_global_batch=${EFFECTIVE_BATCH}"
echo "positive_images=${POSITIVE_TRAIN_IMAGES}; synthetic_negative_images=${SYNTHETIC_NEGATIVE_IMAGES}; total_images=${TOTAL_TRAIN_IMAGES}"
echo "iterations_per_epoch=${ITERS_PER_EPOCH}; target_epochs=${EPOCHS}"
echo "optimizer_peak_lr=5e-5 (launcher tblr=${BASE_LR}, official code multiplies by ac=${ACCUM_STEPS})"
if [[ "${ACCUM_STEPS}" -gt 1 ]]; then
  echo "NOTICE: gradient accumulation fallback selected; effective batch and peak LR remain 32 and 5e-5."
fi

set +e
python scripts/experiment_report.py resume-check \
  --experiment "${EXP_NAME}" \
  --policy "${RESUME_POLICY}" \
  --epochs "${EPOCHS}" \
  --iters-per-epoch "${ITERS_PER_EPOCH}"
RESUME_RC=$?
set -e
if [[ "${RESUME_RC}" -eq 10 ]]; then
  TRAIN_COMMAND="training skipped: checkpoint already reaches requested epochs"
  report_event training_already_complete completed 0
  echo "[$(timestamp)] TRAIN SKIPPED: checkpoint already complete"
  exit 0
fi
if [[ "${RESUME_RC}" -ne 0 ]]; then
  TRAIN_COMMAND="resume policy check"
  report_event resume_refused failed "${RESUME_RC}"
  echo "[$(timestamp)] TRAIN REFUSED by resume policy (exit=${RESUME_RC})" >&2
  exit "${RESUME_RC}"
fi

SEED_ARGS=()
if [[ -n "${SEED:-}" ]]; then
  if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
  fi
  SEED_ARGS=(--seed="${SEED}")
fi

TRAIN_CMD=(
  torchrun
  --standalone
  --nproc-per-node="${NUM_GPUS}"
  train.py
  --depth=24
  --batch_size="${MICRO_BATCH}"
  --ac="${ACCUM_STEPS}"
  --ep="${EPOCHS}"
  --fp16=1
  --tblr="${BASE_LR}"
  --alng=1e-4
  --wpe=0.01
  --wp=0.1
  --sche=lin0
  --twd=0.05
  --twde=0.05
  --wandb_flag=False
  --fuse=0
  --exp_name="${EXP_NAME}"
  --train_manifest_path="data/manifests/lsdir_ffhq10k.txt"
  --degradation_config="dataloader/params_realesrgan.yml"
  --valid_data_path="data/benchmarks/StableSR_testsets/RealSRVal_crop128"
  --cfg_mode="${MODE}"
  --synthetic_negative_fraction="${NEGATIVE_FRACTION_ARG}"
  --synthetic_negative_seed=2025
  --vae_model_path="checkpoints/VQVAE.pth"
  --var_pretrain_path="checkpoints/VARSR_C2I.pth"
  --workers="${WORKERS}"
  "${SEED_ARGS[@]}"
)
printf -v TRAIN_COMMAND '%q ' "${TRAIN_CMD[@]}"

TRAIN_ACTIVE=1
interrupt_training() {
  local exit_code="$1"
  local signal_name="$2"
  if [[ "${TRAIN_ACTIVE}" -eq 1 ]]; then
    echo "[$(timestamp)] TRAIN INTERRUPTED by ${signal_name}; exit=${exit_code}"
    report_event training_interrupted interrupted "${exit_code}" || true
    TRAIN_ACTIVE=0
  fi
  exit "${exit_code}"
}
trap 'interrupt_training 130 SIGINT' INT
trap 'interrupt_training 143 SIGTERM' TERM
trap 'interrupt_training 129 SIGHUP' HUP

report_event training_started running -
echo "[$(timestamp)] torchrun started; terminal output is mirrored to ${TRAIN_LOG}"
set +e
"${TRAIN_CMD[@]}"
TRAIN_RC=$?
set -e
TRAIN_ACTIVE=0
trap - INT TERM HUP

if [[ "${TRAIN_RC}" -eq 0 ]]; then
  report_event training_completed completed 0
  echo "[$(timestamp)] OK: training complete"
  echo "checkpoint: results/${EXP_NAME}/ar-ckpt-best.pth"
  echo "report: ${REPORT_DIR}/REPORT.md"
else
  if [[ "${TRAIN_RC}" -eq 130 || "${TRAIN_RC}" -eq 143 || "${TRAIN_RC}" -eq 129 ]]; then
    TRAIN_STATUS="interrupted"
    TRAIN_EVENT="training_interrupted"
  else
    TRAIN_STATUS="failed"
    TRAIN_EVENT="training_failed"
  fi
  report_event "${TRAIN_EVENT}" "${TRAIN_STATUS}" "${TRAIN_RC}"
  echo "[$(timestamp)] TRAIN ${TRAIN_STATUS}; exit=${TRAIN_RC}; rerun with --resume auto|required" >&2
  exit "${TRAIN_RC}"
fi
