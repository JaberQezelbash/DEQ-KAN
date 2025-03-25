# test.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import sys

from dataset import ShenzhenXRayDataset, compute_metrics_binary
from equilibrium_kan import EquilibriumKANClassifier

def set_seed(seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(root_dataset, model_path):
    set_seed(2024)
    transformations = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    full_dataset = ShenzhenXRayDataset(root_dir=root_dataset, transform=transformations)
    dataset_size = len(full_dataset)
    train_size = int(0.70 * dataset_size)
    val_size = int(0.10 * dataset_size)
    test_size = dataset_size - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    batch_size = 16
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EquilibriumKANClassifier(num_classes=1, hidden_dim=128, num_knots=10,
                                      max_iter=25, tol=1e-4, alpha=0.1, dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    test_loss_total = 0.0
    test_samples = 0
    test_labels_all = []
    test_preds_all = []

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.float().to(device)
            test_logits = model(test_images)
            loss = criterion(test_logits, test_labels)
            bs = test_labels.size(0)
            test_loss_total += loss.item() * bs
            test_samples += bs
            t_probs = torch.sigmoid(test_logits)
            t_preds = (t_probs >= 0.5).long()
            test_labels_all.append(test_labels.long().cpu())
            test_preds_all.append(t_preds.cpu())

    test_loss_avg = test_loss_total / test_samples
    test_labels_cat = torch.cat(test_labels_all, dim=0)
    test_preds_cat = torch.cat(test_preds_all, dim=0)
    test_acc, test_prec, test_rec, test_spec = compute_metrics_binary(test_labels_cat, test_preds_cat)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels_cat.numpy(), test_preds_cat.numpy(), labels=[0,1])
    print(f"Test Loss:       {test_loss_avg:.4f}")
    print(f"Test Accuracy:   {test_acc*100:.2f}%")
    print(f"Test Precision:  {test_prec:.3f}")
    print(f"Test Recall:     {test_rec:.3f}")
    print(f"Test Specificity:{test_spec:.3f}")
    print("Confusion Matrix [Normal=0, Tuberculosis=1]:")
    print(cm)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py <root_dataset_path> <model_path>")
        sys.exit(1)
    root_dataset = sys.argv[1]
    model_path = sys.argv[2]
    main(root_dataset, model_path)
