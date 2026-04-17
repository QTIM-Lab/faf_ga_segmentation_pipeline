#!/bin/bash

FILE_OUT_CLASS="results/classification_results.csv"
OUT_FOLDER="results/ga_segmentations"

CKPT_FILE="models/classification_model_weights.ckpt"
TENSOR_CACHE_DIR='ga_classifier/tensors'
IMAGE_COL="file_path"
N_WORKERS=10
BATCH_SIZE=16
GPU=0

mkdir -p $OUT_FOLDER

echo "Starting Classification of GA"

python ga_classifier/classify.py \
    --file_in    "$FILE_IN" \
    --tensor_cache_dir "$TENSOR_CACHE_DIR" \
    --ckpt_file "$CKPT_FILE" \
    --image_col  "$IMAGE_COL" \
    --file_out   "$FILE_OUT_CLASS" \
    --n_workers  "$N_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --gpu "$GPU"

rm $TENSOR_CACHE_DIR/*.pt

# echo "Starting segmentation of GA"

CUDA_VISIBLE_DEVICES=0 python ga_segmentation/GASegPureInference.py \
  --inference_data_path $FILE_OUT_CLASS \
  --image_col $IMAGE_COL \
  --model_ckpt "models/segmentation_model_weights.pth" \
  --image_size 1024 1024 \
  --batch_size 16 \
  --output_save_path $OUT_FOLDER