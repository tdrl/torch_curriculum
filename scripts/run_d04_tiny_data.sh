#!/bin/zsh

set -e

EPOCHS=3
BATCH_SIZE=25
D_EMBED=128
SEQ_LEN=16
NGRAM_LEN=3

WORK_DIR=/tmp/${USER}/tiny_names_test
[[ -d "${WORK_DIR}" ]] || mkdir -p "${WORK_DIR}"

DATA_DIR=${HOME}/private/data/finnlp_humannames
NAMES_FILE="${DATA_DIR}/human_names_tiny_sample.txt"

# Build toy tokenizer
echo "========== building toy tokenizer =========="
uv run src/torch_playground/build_tokenizer.py \
    --data_file="${NAMES_FILE}" \
    --ngram_len="${NGRAM_LEN}" \
    --output_dir="${WORK_DIR}"

echo "========== training toy model =========="
uv run src/torch_playground/d04_name_seq_learner.py \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --d_embedding="${D_EMBED}" \
    --in_seq_length="${SEQ_LEN}" \
    --out_seq_length="${SEQ_LEN}" \
    --tokenizer_file="${WORK_DIR}/token_dict.n=${NGRAM_LEN}.json" \
    --names_file="${NAMES_FILE}"