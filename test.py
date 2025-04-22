import os
import csv
import torch
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from settings import FEATURES_USED, MODEL_SAVE_PATH, logger, BATCH_SIZE, LEARNING_RATE

def testing(model, criterion, test_loader, device):
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss /= test_total
    test_acc = 100 * test_correct / test_total
    logger.info(f"Test — Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Log to CSV if f1 >= 0.60 for any class
    log_path = "results_log.csv"
    fieldnames = [
        "features", "model", "params", "val_acc", "test_acc", "test_loss",
        "class", "precision", "recall", "f1", "support", "confusion_row"
    ]
    rows_to_log = []
    for label, metrics in report.items():
        if label.isdigit():
            f1 = metrics['f1-score']
            if f1 >= 0.60:
                row = {
                    "features": "+".join(FEATURES_USED),
                    "model": model.__class__.__name__,
                    "params": f"batch_size={BATCH_SIZE},lr={LEARNING_RATE}",
                    "val_acc": "N/A",
                    "test_acc": round(test_acc, 2),
                    "test_loss": round(test_loss, 4),
                    "class": label,
                    "precision": round(metrics['precision'], 4),
                    "recall": round(metrics['recall'], 4),
                    "f1": round(f1, 4),
                    "support": int(metrics['support']),
                    "confusion_row": cm[int(label)].tolist()
                }
                rows_to_log.append(row)

    if rows_to_log:
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_log)

    # Print final metrics
    logger.info("\nClassification Report:")
    for label, metrics in report.items():
        if label.isdigit():
            logger.info(f"Class {label}: F1 = {metrics['f1-score']:.2f}, Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}")

    logger.info("Confusion Matrix:")
    logger.info(cm)
