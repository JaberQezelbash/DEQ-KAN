# train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import sys

from dataset import ShenzhenXRayDataset, compute_metrics_binary
from equilibrium_kan import EquilibriumKANClassifier
from weights_scheduler import weights_init, GradualWarmupScheduler

def set_seed(seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(root_dataset, save_model_path):
    set_seed(2024)
    
    # Data transformations
    transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    full_dataset = ShenzhenXRayDataset(root_dir=root_dataset, transform=transformations)
    dataset_size = len(full_dataset)
    train_size = int(0.70 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Not used here

    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EquilibriumKANClassifier(
        num_classes=1,
        hidden_dim=128,
        num_knots=10,
        max_iter=25,
        tol=1e-4,
        alpha=0.1,
        dropout_p=0.2
    ).to(device)
    model.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    base_lr = 3e-4
    warmup_start_lr = 3e-5
    warmup_epochs = 2

    optimizer = optim.Adam(model.parameters(), lr=warmup_start_lr, weight_decay=1e-5)
    step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = GradualWarmupScheduler(optimizer, warmup_start_lr, base_lr, warmup_epochs, step_scheduler)

    EPOCHS = 20
    patience = 4
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        model.train()
        scheduler.step(epoch)
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                correct = (preds == labels.long()).sum().item()
            bs_curr = labels.size(0)
            running_loss += loss.item() * bs_curr
            running_correct += correct
            total_samples += bs_curr

            if batch_idx % 2 == 0:
                batch_acc = 100.0 * correct / bs_curr
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Train Batch {batch_idx}] Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}% | LR: {current_lr:e}")

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_correct / total_samples

        # Validation loop
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        all_labels_val = []
        all_preds_val = []

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                val_logits = model(val_images)
                vloss = criterion(val_logits, val_labels)
                bs_val = val_labels.size(0)
                val_loss_total += vloss.item() * bs_val
                val_samples += bs_val
                vp_probs = torch.sigmoid(val_logits)
                vp_preds = (vp_probs >= 0.5).long()
                all_labels_val.append(val_labels.long().cpu())
                all_preds_val.append(vp_preds.cpu())

        val_loss_avg = val_loss_total / val_samples
        all_labels_cat = torch.cat(all_labels_val, dim=0)
        all_preds_cat = torch.cat(all_preds_val, dim=0)
        val_acc, val_precision, val_recall, val_specificity = compute_metrics_binary(all_labels_cat, all_preds_cat)

        print(f">> End Epoch {epoch} | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"               | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc*100:.2f}%, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, Specificity: {val_specificity:.3f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), save_model_path)
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <root_dataset_path> <save_model_path>")
        sys.exit(1)
    root_dataset = sys.argv[1]
    save_model_path = sys.argv[2]
    main(root_dataset, save_model_path)
