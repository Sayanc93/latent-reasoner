#! /bin/bash

MODEL_NAME=$1
OUTPUT_FOLDER_NAME=$2

seed_list_aime24=(100 200 300 400 500 600 700 800)
seed_list_aime25=(100 200 300 400 500 600 700 800)
MODEL_TYPE="qwen"


# AIME 24
for seed in ${seed_list_aime24[@]}; do
    bash generate_aime.sh ${MODEL_NAME} ${seed} aime24 ${OUTPUT_FOLDER_NAME} ${MODEL_TYPE}
done
python evaluate_aime.py --modelfolder ${OUTPUT_FOLDER_NAME} --dataset data/aime24.jsonl


# AIME 25
for seed in ${seed_list_aime25[@]}; do
    bash generate_aime.sh ${MODEL_NAME} ${seed} aime25 ${OUTPUT_FOLDER_NAME} ${MODEL_TYPE}
done
python evaluate_aime.py --modelfolder ${OUTPUT_FOLDER_NAME} --dataset data/aime25.jsonl