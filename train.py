import time
import gc
import torch
from tqdm import tqdm
from settings import logger, MODEL_SAVE_PATH

# AMP safe init
if torch.cuda.is_available():
    from torch.amp import autocast, GradScaler
    scaler = GradScaler(device="cuda")
    use_amp = True
else:
    from contextlib import nullcontext
    autocast = nullcontext  # dummy context manager
    scaler = None
    use_amp = False

def training(model, criterion, scheduler, num_epochs, train_loader, val_loader,
             device, optimizer, early_stop_counter, patience, best_val_loss):

    # Optional compile only if CUDA is available
    if torch.cuda.is_available():
        model = torch.compile(model)

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}...")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {time.time() - epoch_start_time:.2f}s. "
                    f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        gc.collect()
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        logger.info(f"Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"  >> New best val loss: {best_val_loss:.4f} — model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"  >> Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                logger.info("  >> Early stopping triggered. Training terminated.")
                break
