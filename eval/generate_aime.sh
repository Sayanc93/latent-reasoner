#!/bin/bash

DIR='pwd'

CHECKPOINT_DIR=$1
TOKENIZER_DIR=$1
seed=$2
DATA_NAME=$3
OUTPUT_FOLDER_NAME=$4
MODEL_TYPE=$5

DATA="data/${DATA_NAME}.jsonl"
#######################################
BSZ=30
TOTAL=30
GPUS=8
OUT_SEQ_LEN=32768
top_p=0.95
temperature=0.6
#######################################

for (( gpu=0; gpu<GPUS; gpu++ )); do
  python inference.py \
    --load "${CHECKPOINT_DIR}" \
    --tokenizer-model "${TOKENIZER_DIR}" \
    --max-output-len "${OUT_SEQ_LEN}" \
    --batch-size "${BSZ}" \
    --temperature "${temperature}" \
    --topp "${top_p}" \
    --tensor-parallel-size 1 \
    --seed "${seed}" \
    --bf16 \
    --model-type "${MODEL_TYPE}" \
    --output-folder "${DIR}/${OUTPUT_FOLDER_NAME}" \
    --datapath "${DATA}" \
    --device-id "${gpu}" &

  seed=$(( seed + 1 ))
done

wait
echo "All GPUs finished."

