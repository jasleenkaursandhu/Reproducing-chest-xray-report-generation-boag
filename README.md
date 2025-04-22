# CS598DLH: Reproducing Baselines for Chest X-Ray Report Generation

This repository contains a reproduction of the paper ["Baselines for Chest X-Ray Report Generation"](https://proceedings.mlr.press/v116/boag20a.html) (Boag et al., 2020) as part of the CS 598 Deep Learning for Healthcare course project.

## Overview

Automatic generation of radiology reports from chest X-ray images holds significant potential for clinical workflow assistance. This project implements and evaluates several baseline methods for this task, focusing on both linguistic quality and clinical accuracy of the generated reports.

## Dataset

This project uses the MIMIC-CXR dataset (Johnson et al., 2019), which is the largest publicly available dataset containing both chest radiographs and their associated free-text reports. The dataset contains:
- 473,057 chest X-ray images
- 206,563 free-text radiology reports
- 63,478 patients

For our experiments, we extract the "findings" section from each report and use it as the ground truth text for training and evaluation. The dataset is split into training and test sets with no patient overlap between the sets. We also filter on the AP views same as the original paper.

## Implemented Models

We implement the following baseline methods for chest X-ray report generation:

1. **Random Retrieval**: Randomly selects a report from the training set.
2. **N-gram Language Models**: Implements unigram, bigram, and trigram language models conditioned on the visual features of the query image.
3. **Nearest Neighbor Retrieval**: Retrieves the report of the most similar training image based on visual features.

## Methodology

### Feature Extraction
- DenseNet121 pre-trained on CheXpert dataset is used to extract 1024-dimensional features from each chest X-ray image.
- These features serve as the basis for both nearest neighbor retrieval and conditioning the language models.

### Report Generation
- **Random Model**: Retrieves a random report from the training set for each test image.
- **N-gram Models**: For each test image, finds the K most similar training images and builds a language model from their associated reports.
- **KNN Model**: For each test image, retrieves the report of the single most similar training image.

### Evaluation
The generated reports are evaluated using two complementary approaches:
1. **Natural Language Generation (NLG) Metrics**: BLEU scores and CIDEr are used to assess linguistic quality.
2. **Clinical Accuracy**: The CheXpert labeler extracts 14 medical observations from the generated reports to measure clinical correctness using F1 scores.

## Results

### NLG Metrics

| Model         | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr  |
|---------------|--------|--------|--------|--------|--------|
| Random (Ours) | 0.2323 | 0.1115 | 0.0553 | 0.0281 | 0.0376 |
| Random (Paper)| 0.2536 | 0.1266 | 0.0717 | 0.0436 | 0.0462 |
| 3-gram (Ours) | 0.1990 | 0.0955 | 0.0499 | 0.0254 | 0.0228 |
| 3-gram (Paper)| 0.2258 | 0.1172 | 0.0687 | 0.0418 | 0.0185 |
| KNN (Ours)    | 0.2433 | 0.1164 | 0.0617 | 0.0357 | 0.0721 |
| KNN (Paper)   | 0.2807 | 0.1500 | 0.0905 | 0.0585 | 0.0921 |

### Clinical Metrics (CheXpert F1 Scores)

| Model         | Macro F1 |
|---------------|----------|
| Random (Ours) | 0.1823   |
| Random (Paper)| 0.1480   |
| 3-gram (Ours) | 0.1640   |
| 3-gram (Paper)| 0.1850   |
| KNN (Ours)    | 0.1806   |
| KNN (Paper)   | 0.2580   |

## Setup and Usage

### Prerequisites
- Python 3.6+
- PyTorch
- Pandas, NumPy, Matplotlib
- Scikit-learn
- PyDicom
- Docker (for running the CheXpert labeler)

### Data Preparation
1. Access to the MIMIC-CXR dataset is required (not included in this repository due to data usage agreements).
2. Preprocess the dataset using `01_mimic_cxr_preprocessing.ipynb`.
3. Extract image features using `extract_densenet_features.ipynb`.

### Model Training and Evaluation
1. Generate reports using the individual model notebooks:
   - `random.ipynb`
   - `1gram.ipynb`, `2gram.ipynb`, `3gram.ipynb`
   - `knn.ipynb`
2. Process generated reports through the CheXpert labeler using Docker:
   ```bash
   docker run --platform linux/amd64 -v /path/to/project:/data uwizeye2/chexpert-labeler:amd64 python label.py --reports_path /data/model_headerless.csv --output_path /data/output/labeled_model.csv --verbose