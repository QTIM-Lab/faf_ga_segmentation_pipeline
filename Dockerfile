FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libjpeg62-turbo && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/ga_classifier/tensors /app/ga_segmentation/predictions /app/results

CMD ["bash", "run_pipeline.sh"]
