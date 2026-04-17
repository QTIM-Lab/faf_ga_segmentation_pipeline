FAF GA Classification Pipeline
================================

OVERVIEW
--------
This pipeline classifies Fundus Autofluorescence (FAF) images for the presence
of Geographic Atrophy (GA) using a ResNet50-based binary classifier. It outputs
a probability score per image, where higher values indicate a greater likelihood
of GA being present.

A probability >= 0.5 is used as the default threshold for GA classification.


DIRECTORY STRUCTURE
-------------------
faf_ga_segmentation_pipeline/
└── models/
    └── ga_classifier/
        ├── classify.py            # Main inference script
        ├── run.sh                 # Shell script to run inference
        ├── FAF_example.csv        # Example input CSV
        ├── FAF_example_results.csv  # Example output CSV
        ├── weights/
        │   └── faf_model_weights.ckpt  # Model checkpoint
        └── tensors/               # Auto-generated tensor cache directory


INPUT FORMAT
------------
The input is a CSV file with at least one column containing absolute paths to
FAF images (.j2k or other PIL-readable formats). The column name is specified
via the --image_col argument.

Example:
    file_path
    /path/to/image1.j2k
    /path/to/image2.j2k


OUTPUT FORMAT
-------------
A CSV file with two columns:
    file_path       - The original image path
    ga_probability  - GA probability score between 0 and 1 (NaN if image missing)


USAGE
-----
Run from the /faf_ga_segmentation_pipeline directory:

    bash models/ga_classifier/run.sh

The shell script contains the following configurable variables:

    FILE_IN          Path to input CSV (relative to run directory)
    CKPT_FILE        Path to model checkpoint file
    TENSOR_CACHE_DIR Directory to store cached image tensors (speeds up re-runs)
    IMAGE_COL        Name of the column in FILE_IN containing image paths
    FILE_OUT         Path to save output CSV
    N_WORKERS        Number of parallel workers for image caching (default: 32)
    BATCH_SIZE       Inference batch size (default: 16)
    GPU              GPU device index to use (default: 0)


TENSOR CACHING
--------------
On first run, images are preprocessed and cached as .pt tensor files in
TENSOR_CACHE_DIR. Subsequent runs will skip caching and load tensors directly,
making re-runs significantly faster. If you change the input images, delete
the tensors directory to force re-caching.


MISSING IMAGES
--------------
If an image path does not exist on disk, it will be skipped and its
ga_probability will be recorded as NaN in the output CSV. A warning count
is printed at the end of the run.


REQUIREMENTS
------------
See requirements.txt for full dependency list. Key dependencies:
    torch
    torchvision
    lightning
    pandas
    numpy
    Pillow
    tqdm
    opencv-python


NOTES
-----
- The model was trained on 512x512 FAF images. Input images are automatically
  resized to this resolution during preprocessing.
- The classifier uses a ResNet50 backbone with ImageNet pretraining, fine-tuned
  for binary GA classification.
- GPU is strongly recommended. CPU inference is supported but will be slow.