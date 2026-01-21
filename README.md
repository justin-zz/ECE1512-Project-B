# Histopathology Image Classification with ABMIL
![MIL Pipeline](/images/pipeline.jpg)   
## Project Overview
This repository implements an Attention-Based Multiple Instance Learning (ABMIL) model for histopathology whole slide image (WSI) classification. The model processes image patches and aggregates them with attention mechanisms for slide-level predictions.
The system is designed for binary/multi-class classification of histopathology slides by:

1. Extracting features from individual image patches
2. Weighing patch importance using attention mechanisms
3. Aggregating features for slide-level classification
4. 
![MIL Pipeline](/images/architecture.jpg)   

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

```bash
# Clone this repository
git clone https://github.com/justin-zz/ECE1512-Project-B.git
cd ECE1512-Project-B
```

### Dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn pandas pyyaml torchmetrics
```

### Training
```bash
# Change parameters inside main.py (dataset, epochs, etc...)
python main.py
```

### Visualization
```bash
python create_plots.py
```
