#!/usr/bin/env bash
set -euo pipefail

INPUT=""
PARAMS=""
EXTRA=""
OUTPUT_DIR=""
THREADS=""

RUNTIME_BIN="python"
APP_SCRIPT="/app/run_tool.py"

usage() {
  echo "Usage: /app/run_tool.sh --input /io/expression.tsv --params /io/params.json --extra /io/extra --output-dir /io/out --threads 8"
}

die() {
  echo "Error: $*" >&2
  usage >&2
  exit 1
}

require_value() {
  local arg="$1"
  local value="${2-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    die "Missing value for ${arg}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      require_value "$1" "${2-}"
      INPUT="$2"
      shift 2
      ;;
    --params)
      require_value "$1" "${2-}"
      PARAMS="$2"
      shift 2
      ;;
    --extra)
      require_value "$1" "${2-}"
      EXTRA="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --threads)
      require_value "$1" "${2-}"
      THREADS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -z "${INPUT}" || -z "${PARAMS}" || -z "${EXTRA}" || -z "${OUTPUT_DIR}" || -z "${THREADS}" ]]; then
  die "Missing required arguments."
fi

if ! [[ "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  die "--threads must be a positive integer."
fi

command -v "${RUNTIME_BIN}" >/dev/null 2>&1 || die "'${RUNTIME_BIN}' is not available in PATH."
[[ -f "${APP_SCRIPT}" ]] || die "Internal error: ${APP_SCRIPT} not found."

mkdir -p "${OUTPUT_DIR}"

# Set native-library limits before the interpreter initializes BLAS/OpenMP.
# Worker-based methods use one native thread per worker.
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"
export BLIS_NUM_THREADS="1"
export VECLIB_MAXIMUM_THREADS="1"

exec "${RUNTIME_BIN}" "${APP_SCRIPT}" \
  --input "${INPUT}" \
  --params "${PARAMS}" \
  --extra "${EXTRA}" \
  --output-dir "${OUTPUT_DIR}" \
  --threads "${THREADS}"
