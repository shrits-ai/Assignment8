# config.py
import torch
# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 10

# Dataset configuration
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)  # CIFAR10 dataset mean
CIFAR_STD = (0.2470, 0.2435, 0.2616)   # CIFAR10 dataset std