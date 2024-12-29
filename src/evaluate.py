# evaluate.py

import torch
from tqdm import tqdm
from config import DEVICE

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=True)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update running loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar description
            progress_bar.set_postfix(loss=(running_loss / len(dataloader)), acc=100. * correct / total)

    return running_loss / len(dataloader), 100. * correct / total

