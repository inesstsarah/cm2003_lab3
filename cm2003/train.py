import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def run_batch(model, data, target, criterion, optimizer, is_training=True):
    """

    Run a single batch through the model.

    Complete the code between ---CODE START HERE--- and ---CODE END HERE---:
    1. Set the model to train or eval mode based on is_training
    2. Zero the optimizer gradients if training
    3. Forward pass through the model
    4. Compute the loss
    5. If training, perform backward pass and optimizer step
    """
    # ---CODE START HERE---
    # Hint: MedMNIST labels have shape (batch_size, 1), while nn.CrossEntropyLoss expects
    # targets of shape (batch_size,), so use target.reshape(-1) when computing the loss.
    loss = None
    criterion = nn.CrossEntropyLoss()

    if is_training:
        mode = "train"
        optimizer.zero_grad() # Zeroing optimizer gradients
        # Forward pass through the model
        outputs = model(data)
        # Compute loss
        # Compute the loss and its gradients
        loss = criterion(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    else:
        mode = "eval"
        outputs = model(data)
        # Compute loss
        # Compute the loss and its gradients
        loss = criterion(outputs, target)
    
    


    # ---CODE END HERE---

    return loss.item()

def check_accuracy(model, loader, criterion, device):
    """
    Check accuracy of our trained model given a loader and a model

    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on

    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    total_loss = 0.0
    num_batches = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape

            # Forward pass
            scores = model(x)
            # Compute loss
            loss = criterion(scores, y.reshape(-1))

            _, predictions = scores.max(1)


            # Check how many we got correct
            num_correct += (predictions.reshape(-1) == y.reshape(-1)).sum().item()
            total_loss += loss.item()
            # Keep track of number of samples
            num_samples += predictions.shape[0]
            num_batches += 1
    model.train()
    return total_loss / num_batches, num_correct / num_samples

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Complete the code between ---CODE START HERE--- and ---CODE END HERE---:
    1. Move data and target to the specified device
    2. Call run_batch with appropriate parameters
    3. Add the batch loss to total_loss
    """
    model.train()
    total_loss = 0.0

    for data, target in loader:
        # ---CODE START HERE---

        # Move data to the specified device
        data = data.to(device)
        # Call run_batch
        batch_loss = run_batch(model, data, target, criterion, optimizer)
        #Add batch loss
        total_loss+=batch_loss



        # ---CODE END HERE---

    avg_loss = total_loss / len(loader)
    return avg_loss

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, check_every=1):
    """
    Train the model for multiple epochs.

    Complete the code between ---CODE START HERE--- and ---CODE END HERE---:
    1. Call train_epoch with appropriate parameters
    2. Store the returned average training loss in train_loss
    """
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        # ---CODE START HERE---
        train_loss = train_epoch(model,train_loader,criterion, optimizer, device) 

        # ---CODE END HERE---

        results['train_loss'].append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}:")

        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        if epoch % check_every == 0:
            val_loss, val_accuracy = check_accuracy(model, val_loader, criterion, device)
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_accuracy)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


    return model, results
