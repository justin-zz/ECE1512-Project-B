# Histopathology Image Classification with ABMIL

## Project Overview
This repository implements an Attention-Based Multiple Instance Learning (ABMIL) model for histopathology whole slide image (WSI) classification. The model processes image patches and aggregates them with attention mechanisms for slide-level predictions.
The system is designed for binary/multi-class classification of histopathology slides by:

1. Extracting features from individual image patches
2. Weighing patch importance using attention mechanisms
3. Aggregating features for slide-level classification

## Features

### Training Pipeline

- Multi-GPU support with automatic device detection
- Comprehensive metrics tracking (Accuracy, AUC, F1)
- Learning rate scheduling with warmup + cosine annealing
- Model checkpointing and early stopping
- Detailed logging and visualization

### Analysis Tools
- Training curves (loss, accuracy, AUC, F1)
- Model parameter analysis and visualization
- Performance comparison across configurations
- Metric correlation analysis
- Automated report generation (HTML/CSV/PDF)

### Performance Metrics
- Accuracy: Overall classification correctness
- AUC-ROC: Area under the ROC curve
- F1 Score: Harmonic mean of precision and recall
- Loss: MultiMargin loss for training

## Quick Start

### Installation
#### Clone repository
#### Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib seaborn pandas pyyaml torchmetrics

### Training
Change parameters inside main.py (dataset, epochs, etc...)
python main.py

### Visualization
python create_plots.py

```bash
# Dependencies will be installed automatically when running the script
# Manual installation if needed:
pip install torch torchvision matplotlib numpy tqdm scipy scikit-learn
