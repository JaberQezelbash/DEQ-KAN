# dataset.py
import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import torch

class ShenzhenXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        normal_paths = glob.glob(os.path.join(root_dir, 'Normal', '*.*'))
        tb_paths     = glob.glob(os.path.join(root_dir, 'Tuberculosis', '*.*'))

        self.samples = []
        for path in normal_paths:
            self.samples.append((path, 0))
        for path in tb_paths:
            self.samples.append((path, 1))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def compute_metrics_binary(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    accuracy  = np.mean(y_true_np == y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall    = recall_score(y_true_np, y_pred_np, zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-9)
    return accuracy, precision, recall, specificity

if __name__ == "__main__":
    # Quick test for dataset loading
    dataset = ShenzhenXRayDataset(root_dir="path/to/dataset", transform=transforms.ToTensor())
    print("Dataset length:", len(dataset))
