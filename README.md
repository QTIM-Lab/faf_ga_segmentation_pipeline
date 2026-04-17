# FAF GA Segmentation Pipeline

This repository contains a pipeline for classifying and segmenting geographic atrophy (GA) from FAF images.

The project includes:
- `ga_classifier/`: FAF image classification model and inference script
- `ga_segmentation/`: SAM-based segmentation inference using a pretrained model
- `models/`: pretrained model weights for classification and segmentation
- `Dockerfile` and `docker-compose.yml`: containerized environment for reproducible execution
- `run_pipeline.sh`: wrapper script that runs classification followed by segmentation

## Docker usage

Build the container:

```bash
docker compose up --build
```

## Repository structure

Important files to download separately:
- `models`: folder to create to store models
- `models/classification_model_weights.ckpt`: classification model weights
- `models/segmentation_model_weights.pth`: segmentation model weights
- `imgs/`: folder where images can be placed - if images are somewhere else, mount the location in the docker-compose file

Important input file:
- `FAF_files.csv`: example input CSV file (this is the default name, change it in the docker-compose)

Other files:
- `ga_classifier/classify.py`: main classification script
- `ga_classifier/classify.sh`: sample classifier launch script
- `ga_classifier/tensors/`: cache directory for preprocessed image tensors
- `ga_segmentation/GASegPureInference.py`: segmentation inference script
- `ga_segmentation/segment_ga.sh`: sample segmentation launcher
- `run_pipeline.sh`: example full pipeline invocation
- `results/`: expected output directory for segmentation masks

## Notes

- The classification script caches preprocessed image tensors under `ga_classifier/tensors/`.
- Segmentation outputs are written to `results/ga_segmentations/` by default.
- If your environment does not have a GPU, update the `--gpu` argument in `run_pipeline.sh` and the corresponding script variables to use CPU-only execution.
- Verify that the expected model weights exist in `models/` before running inference.
