import os
import torch

# Paths
BASE_PATH = 'dataset'
SAVE_DIR = 'checkpoints/model'
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_EPOCHS = 1
N_SPLITS = 3
ALPHA = 0.7
BETA = 0.3
NUM_CLASSES = 3

# Augmentation
LATENT_NOISE_SIGMA = 0.05