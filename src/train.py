from tqdm import tqdm
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    processed = 0

    # Initialize progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()
        processed += len(inputs)
        # Update progress bar description
        progress_bar.set_postfix(loss=(running_loss / (batch_idx + 1)))

    return running_loss / len(dataloader), (( correct / processed ) * 100)