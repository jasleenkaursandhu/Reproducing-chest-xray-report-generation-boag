# CS598DLH: Reproducing "Baselines for Chest X-Ray Report Generation"

This repository contains a reproduction of the paper ["Baselines for Chest X-Ray Report Generation"](https://proceedings.mlr.press/v116/boag20a/boag20a.pdf) (Boag et al., 2020) as part of the CS 598 Deep Learning for Healthcare course project at the University of Illinois at Urbana-Champaign.

## Authors
- Jasleen Kaur Sandhu (jasleen3)
- Simeon Uwizeye (uwizeye2)

## Overview

This project reproduces and extends the work of Boag et al. (2020) in establishing baseline methods for automatically generating radiology reports from chest X-ray images. Our work validates the key finding that simpler retrieval-based approaches can perform competitively with more complex neural models while providing significant computational efficiency benefits.

## Table of Contents
- [Dataset](#dataset)
- [Implemented Models](#implemented-models)
- [Extended Experiments](#extended-experiments)
- [Methodology](#methodology)
- [Results](#results)
- [Docker Setup for CheXpert Labeler](#docker-setup-for-chexpert-labeler)

## Dataset

We use a subset of the MIMIC-CXR dataset (Johnson et al., 2019), focusing on Anteroposterior (AP) view X-rays:

- **Total Dataset**: 377,110 chest X-ray images
- **Filtered Dataset**: 6,048 AP view images with corresponding reports
- **Train/Test Split**: 70/30 patient-level split
  - Training: 4,291 images (2,273 patients)
  - Test: 1,757 images (975 patients)

### Data Filtering Process
1. Filter for AP view X-rays using DICOM metadata
2. Verify presence of "findings" section in reports
3. Create patient-level train/test split
4. Extract "findings" section as ground truth text

## Implemented Models

### 1. Random Baseline
- Randomly selects a report from the training set for each test image
- Serves as our most basic baseline

### 2. N-gram Language Models
Three variants of conditional language models:
- **Unigram (1-gram)**: Words sampled based on frequency distribution
- **Bigram (2-gram)**: Words conditioned on the previous word
- **Trigram (3-gram)**: Words conditioned on the previous two words

For each test image:
1. Find 100 most similar training images (by default)
2. Build n-gram model from their reports
3. Generate new report by sampling from the model

### 3. K-Nearest Neighbor (KNN)
- Retrieves the report of the most visually similar training image
- Uses cosine similarity between DenseNet features
- Simple yet effective retrieval-based approach

### 4. CNN-RNN Neural Network
- **Encoder**: DenseNet121 pre-trained on ImageNet
- **Decoder**: LSTM with 512 hidden units
- **Training**: Teacher forcing with gradual decay
- **Inference**: Beam search with width 3-4
- **Enhancements**: Repetition penalty (factor 1.2)

## Extended Experiments

We extended the original work by experimenting with different neighbor counts for n-gram models:

### Neighbor Count Variations
Tested n-gram models with 10, 50, 100, and 200 nearest neighbors:

### Extended Experiments: N-gram Models with Different Neighbor Counts

#### NLG metrics for n-gram models with different neighbor counts

| Model  | Neighbors | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|--------|-----------|--------|--------|--------|--------|-------|
| 1-gram | 10        | 0.1843 | 0.0291 | 0.0042 | 0.0010 | 0.1378 |
| 1-gram | 50        | 0.1783 | 0.0279 | 0.0035 | 0.0000 | 0.1206 |
| 1-gram | 100       | 0.1833 | 0.0285 | 0.0021 | 0.0000 | 0.1247 |
| 1-gram | 200       | 0.1887 | 0.0300 | 0.0027 | 0.0000 | 0.1184 |
| 2-gram | 10        | 0.1952 | 0.0955 | 0.0461 | 0.0231 | 0.1732 |
| 2-gram | 50        | 0.1981 | 0.0964 | 0.0444 | 0.0200 | 0.2079 |
| 2-gram | 100       | 0.1931 | 0.0936 | 0.0428 | 0.0195 | 0.2053 |
| 2-gram | 200       | 0.1922 | 0.0933 | 0.0433 | 0.0198 | 0.1970 |
| 3-gram | 10        | 0.2052 | 0.1012 | 0.0544 | 0.0306 | 0.3015 |
| 3-gram | 50        | 0.2067 | 0.1011 | 0.0544 | 0.0312 | 0.2330 |
| 3-gram | 100       | 0.1979 | 0.0968 | 0.0525 | 0.0295 | 0.2874 |
| 3-gram | 200       | 0.1994 | 0.0979 | 0.0521 | 0.0286 | 0.2311 |

#### Clinical accuracy metrics for n-gram models with different neighbor counts

| Model  | Neighbors | Macro Accuracy | Macro Precision | Macro F1 | Micro F1 |
|--------|-----------|----------------|-----------------|----------|----------|
| 1-gram | 10        | 0.5932         | 0.1704          | 0.1634   | 0.2533   |
| 1-gram | 50        | 0.5876         | 0.1645          | 0.1610   | 0.2519   |
| 1-gram | 100       | 0.5915         | 0.1682          | 0.1627   | 0.2551   |
| 1-gram | 200       | 0.5880         | 0.1658          | 0.1643   | 0.2570   |
| 2-gram | 10        | 0.6043         | 0.1853          | 0.1724   | 0.2966   |
| 2-gram | 50        | 0.5962         | 0.1765          | 0.1653   | 0.2827   |
| 2-gram | 100       | 0.5923         | 0.1832          | 0.1734   | 0.2772   |
| 2-gram | 200       | 0.6004         | 0.1902          | 0.1803   | 0.2970   |
| 3-gram | 10        | 0.6040         | 0.1887          | 0.1735   | 0.2987   |
| 3-gram | 50        | 0.6033         | 0.1741          | 0.1629   | 0.2990   |
| 3-gram | 100       | 0.5996         | 0.1839          | 0.1692   | 0.2906   |
| 3-gram | 200       | 0.5989         | 0.1790          | 0.1692   | 0.2869   |

Key Finding: Performance generally increases with more neighbors up to 100, then plateaus or slightly decreases at 200.

## Methodology

### Feature Extraction
- DenseNet121 pre-trained on ImageNet
- Global average pooling → 1024-dimensional features
- Cosine similarity for image comparison

### Evaluation Metrics
1. **Natural Language Generation (NLG) Metrics**:
   - BLEU (1-4): N-gram precision
   - CIDEr: Consensus-based evaluation

2. **Clinical Accuracy Metrics**:
   - CheXpert labeler extracts 14 medical observations
   - Macro/Micro F1 scores for clinical correctness

## Results

### NLG Metrics Comparison

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr (×10) |
|-------|--------|--------|--------|--------|-------------|
| Random | 0.232 | 0.111 | 0.057 | 0.031 | 0.470 |
| 1-gram | 0.196 | 0.031 | 0.003 | 0.000 | 0.130 |
| 2-gram | 0.196 | 0.095 | 0.043 | 0.020 | 0.186 |
| 3-gram | 0.198 | 0.099 | 0.055 | 0.031 | 0.275 |
| KNN | 0.244 | 0.121 | 0.066 | 0.038 | 0.450 |

### Clinical Accuracy Metrics

| Model | Macro Accuracy | Macro Precision | Macro F1 | Micro F1 |
|-------|----------------|-----------------|----------|----------|
| Random | 0.5891 | 0.1658 | 0.1689 | 0.3065 |
| 1-gram | 0.5894 | 0.1698 | 0.1643 | 0.2592 |
| 2-gram | 0.5978 | 0.1843 | 0.1756 | 0.2933 |
| 3-gram | 0.5991 | 0.1769 | 0.1626 | 0.2841 |
| KNN | 0.6162 | 0.2129 | 0.2086 | 0.3541 |


## Setup and Usage

### Prerequisites

python>=3.8
pytorch>=1.12.0
tensorflow>=2.10.0
numpy
pandas
pydicom
scikit-learn
matplotlib
tqdm
docker

## Docker Setup for CheXpert Labeler

# Pull our custom Docker image
docker pull uwizeye2/chexpert-labeler:amd64

# Prepare input files
cp output/model_predictions.tsv output/model_headerless.csv
sed -i '1d' output/model_headerless.csv

# Run the labeler
docker run --platform linux/amd64 -v $(pwd):/data uwizeye2/chexpert-labeler:amd64 \
    python label.py --reports_path /data/output/model_headerless.csv \
    --output_path /data/output/labeled_model.csv --verbose
