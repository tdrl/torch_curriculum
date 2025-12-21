#!/bin/zsh

set -e

typeset -A arg_vals
zparseopts -D -E -A arg_vals h -help -small -full

for k v in "${(@kv)arg_vals}"; do
    echo "[${k}] -> '${v}'"
done

if [[ -v arg_vals[-h] || -v arg_vals[--help] ]]; then
    echo "Usage: $0 [--small|--full]"
    exit 0
fi
if [[ -v arg_vals[--small] && -v arg_vals[--full] ]]; then
    echo "Sorry - you have to pick either small or full."
    exit 1
fi

DATA_DIR=${HOME}/private/data/finnlp_humannames

if [[ -v arg_vals[--small] ]]; then
    EPOCHS=3
    BATCH_SIZE=25
    D_EMBED=128
    SEQ_LEN=16
    NGRAM_LEN=3

    WORK_DIR=/tmp/${USER}/tiny_names_test
    NAMES_FILE="${DATA_DIR}/human_names_tiny_sample.txt"
    echo "Running in small / test mode:"
    echo "EPOCHS=${EPOCHS}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "D_EMBED=${D_EMBED}"
    echo "SEQ_LEN=${SEQ_LEN}"
    echo "NGRAM_LEN=${NGRAM_LEN}"
    echo "WORK_DIR=${WORK_DIR}"
    echo "NAMES_FILE=${NAMES_FILE}"

elif [[ -v arg_vals[--full] ]]; then
    EPOCHS=10
    BATCH_SIZE=50
    D_EMBED=128
    SEQ_LEN=16
    NGRAM_LEN=3

    WORK_DIR=${HOME}/private/experiments/name_seq_learning
    NAMES_FILE="${DATA_DIR}/human_names_list.txt"
    echo "Running in full / prod mode:"
    echo "EPOCHS=${EPOCHS}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "D_EMBED=${D_EMBED}"
    echo "SEQ_LEN=${SEQ_LEN}"
    echo "NGRAM_LEN=${NGRAM_LEN}"
    echo "WORK_DIR=${WORK_DIR}"
    echo "NAMES_FILE=${NAMES_FILE}"
else
    echo "Need to specify either --small or --full"
    exit 1
fi

[[ -d "${WORK_DIR}" ]] || mkdir -p "${WORK_DIR}"

# Build full tokenizer. Could shortcut this if it
# already exits but, honestly, this is a negligible
# fraction of the total runtime and there's a marginal
# chance of the token dict becoming disconnected from
# the model if we did shortcut. So just suck it up and
# regenerate every time.
echo "========== building tokenizer =========="
uv run src/torch_playground/build_tokenizer.py \
    --data_file="${NAMES_FILE}" \
    --ngram_len="${NGRAM_LEN}" \
    --output_dir="${WORK_DIR}"

echo "========== training model =========="
uv run src/torch_playground/d04_name_seq_learner.py \
    --epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --d_embedding="${D_EMBED}" \
    --in_seq_length="${SEQ_LEN}" \
    --out_seq_length="${SEQ_LEN}" \
    --tokenizer_file="${WORK_DIR}/token_dict.n=${NGRAM_LEN}.json" \
    --names_file="${NAMES_FILE}" \
    --output_dir="${WORK_DIR}"
