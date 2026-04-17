#!/bin/bash
# THIS SCRIPT ASSUMES YOU ARE RUNNING THIS FROM THE /faf_ga_segmentation_pipeline DIRECTORY

FILE_IN=$1
CKPT_FILE="models/classification_model_weights.ckpt"
TENSOR_CACHE_DIR='ga_classifier/tensors'
IMAGE_COL="file_path"
FILE_OUT=$2
N_WORKERS=32
BATCH_SIZE=16
GPU=0

python ga_classifier/classify.py \
    --file_in    "$FILE_IN" \
    --tensor_cache_dir "$TENSOR_CACHE_DIR" \
    --ckpt_file "$CKPT_FILE" \
    --image_col  "$IMAGE_COL" \
    --file_out   "$FILE_OUT" \
    --n_workers  "$N_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --gpu "$GPU"