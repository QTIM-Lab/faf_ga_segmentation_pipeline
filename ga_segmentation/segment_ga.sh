#!/usr/bin/env bash
# THIS SCRIPT ASSUMES YOU ARE RUNNING THIS FROM THE BASE OF THE /faf_ga_segmentation_pipeline DIRECTORY

FILE=$1
OUT_FOLDER=$2

python ga_segmentation/GASegPureInference.py \
  --inference_data_path $FILE \
  --image_col "file_path" \
  --model_ckpt "../models/segmentation_model_weights.pth" \
  --image_size 1024 1024 \
  --batch_size 16 \
  --output_save_path $OUT_FOLDER