# train.py
import os
import warnings

# Disable version checking in Albumentations
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

# Suppress specific UserWarning from Albumentations
warnings.filterwarnings("ignore", message="Error fetching version info")

import torch
import torch.optim as optim
from model import CIFAR10Net
from data_preprocessing import get_data_loaders
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE
from evaluate import validate
from train import train
from torchsummary import summary
from checkpoint import save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import OneCycleLR

if __name__ == '__main__':
    # Initialize model
    model = CIFAR10Net().to(DEVICE)
    summary(model, input_size=(3, 32, 32))
    trainloader, testloader = get_data_loaders(batch_size=BATCH_SIZE)

    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE )  # Optimizer
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=EPOCHS)

    checkpoint_path = "checkpoint.pth"
    best_loss = float('inf')

    # Load checkpoint if it exists to resume training
    try:
        model, optimizer, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    # Training and validation loop
    for epoch in range( 1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss,train_acc = train(model, trainloader, criterion, optimizer, DEVICE)
        scheduler.step()
        val_loss, val_acc = validate(model, testloader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Train Acc : {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        
        # Save the best model (optional, you can adjust this)
        if train_loss < best_loss:
            best_loss = train_loss
            # Save checkpoint after each epoch (or based on your condition)
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)