# Attention-Guided Joint Segmentation & Classification of Breast Ultrasound with Latent Feature Augmentation

This repository contains the complete implementation and results of Thesis Phase 2, focused on developing a high-performance Joint Classification and Segmentation architecture for breast ultrsond image.
The project documents the transition from a baseline model to a highly optimized final architecture designed to handle diagnostic complexity and data scarcity in breast cancer imaging.

## Repository Structure
Dual_Aug_Net/
├── data/dataset/                # Data subfolders: benign/, malignant/, normal/
├── models/
│   ├── advanced_modules.py      # FFT Bottleneck & Latent Augmentor logic
│   ├── decoders.py              # UNet, UNet++, and DeepLabV3+ architectures
│   ├── generic_model.py         # Flexible model for encoder benchmarking
│   └── class_imbalance.py       # Final Advanced Model (B6 + FFT + LatentAug)
├── utils/
│   ├── dataset.py               # Medical dataset loading & augmentation
│   ├── losses.py                # DiceFocal and Weighted Cross-Entropy losses
│   └── metrics.py               # DSC, NSD, HD95, AUC, and F1 calculations
├── config.py                    # Global hyperparameters and path settings
├── run_experiments.py           # Experiment 1: Benchmarking 20+ Encoders
├── run_experiments_decoders.py  # Experiment 2: Benchmarking Decoders
├── run_class_experiments.py     # Experiment 3: Final Advanced Training
└── predict.py                   # CLI Tool for single-image inference

## Installation & Setup

### Clone and Install:

```bash
git clone https://github.com/Nimalesh/Dual_Aug_Net.git
cd Dual_Aug_Net
pip install -r requirements.txt
```

### Fix NumPy 2.x Compatibility:
```
pip install "numpy<2"
```

### Data Preparation
Organise as follow
```
data/dataset/benign/*.png
data/dataset/malignant/*.png
data/dataset/normal/*.png
```
Segmentation masks must be in the same folders as imageName_mask.png.

### Thesis Phase 2: Experimental Journey
The implementation follows a rigorous three-stage experimental protocol to arrive at the final optimized architecture.
#### Experiment 1: Encoder (run_experiments.py)

To establish a powerful feature extractor, this phase involved benchmarking over 20 different encoders—including ResNet (18, 50, 101), EfficientNet (B0–B5), MobileNet, and ShuffleNet. Each was evaluated using 3-Fold Cross-Validation to find the backbone with the highest macro-F1 and feature extraction capability.

#### Experiment 2: Decoder Selection (run_experiments_decoders.py)

Using the top-performing encoder (EfficientNet-B6), this phase compared three major decoder architectures: UNet, UNet++ (Nested UNet), and DeepLabV3+. The goal was to determine which decoder best reconstructed spatial resolution and handled fine-grained boundary segmentation of lesions.

#### Experiment 3: Attention moduls:

A critical part of the thesis involved selecting the right attention mechanism to prioritize relevant clinical features. CBAM (Convolutional Block Attention Module) was selected and integrated into the bottleneck.
#### Channel Attention: Prioritizes "what" features (lesion characteristics) are important.
#### Spatial Attention: Prioritizes "where" the lesion is located within the ultrasound frame.
This selection was instrumental in reducing noise from non-tissue areas in the ultrasound scans.

### Experiment 3: Final Architecture & Class Imbalance (run_class_experiments.py)

The final phase implemented the optimized joint architecture. This involved integrating a Dual-Domain Bottleneck (Spatial + Frequency FFT) to capture global texture patterns. To address class imbalance, Latent Augmentation (Gaussian Noise & Intra-class MixUp) was implemented in the bottleneck, specifically targeting minority classes (Normal and Malignant) to improve generalization.


### Final Architecture Selection
Based on the results of the thesis experiments, the final model consists of:
#### Encoder: EfficientNet-B6 (Pre-trained)
#### Bottleneck: Spatial-Frequency Dual Domain Processing + CBAM Attention
#### Decoder: UNet++ (Nested Skip-Connections)
#### Optimization: Latent Space Feature Augmentation for imbalance handling.


### Running Experiments
#### 1. Encoder 
20+ different backbones (ResNets, EfficientNets, MobileNets, etc.) was experimented to find the best feature extractor for data:

```bash
python run_experiments.py
```
#### 2. Decoder Architecture Comparison
Compare UNet, UNet++, and DeepLabV3+ performance using a fixed EfficientNet-B6 backbone:

```bash
python run_experiments_decoders.py
```

#### 3. Class imbalance handling
Train the final model incorporating EfficientNet-B6, UNet++, Dual-Domain (FFT) Bottleneck, and Latent Augmentation:

bash
```
python run_class_experiments.py
```
The metrics and epoch-wise logs will be saved in csv.

### Main Model Inference
To run diagnostic prediction and segmentation on a single test image:

#### Run on GPU:

```bash

python predict.py --image path/to/your/test_image.png --gpu
```
#### Run on CPU:

```bash
python predict.py --image path/to/your/test_image.png
```
The script will output the classification probabilities and save a visualization named prediction_output.png.

Evaluation Metrics
The pipeline evaluates models using standard medical imaging benchmarks:

Segmentation: Dice Similarity Coefficient (DSC), Normalized Surface Dice (NSD), and Hausdorff Distance 95% (HD95)

Classification: Area Under the Curve (AUC) and Macro-averaged F1-Score

### Troubleshooting & Tips
Memory Management: EfficientNet-B6 is very large. If your Mac crashes, ensure ``` BATCH_SIZE = 2 ``` in config.py

Corrupted Weights: If you see RuntimeError: invalid hash value, your internet interrupted the weight download. Fix it with:

```bash
rm -rf ~/.cache/torch/hub/checkpoints/*
```

