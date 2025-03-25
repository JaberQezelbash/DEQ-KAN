# DEQ-KAN
DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification



<img width="600" alt="kan_plot" src="https://github.com/JaberQezelbash/DEQ-KAN/blob/main/assets/DEQ-KAN.svg">


This repository extends the original Kolmogorov-Arnold Networks (KANs) concept to include our new method, **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks**, as presented in our paper:

> **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**  
> *by Jaber Qezelbash-Chamak*.  
> [[Paper Link]](https://github.com/JaberQezelbash/DEQ-KAN) (Implementation Code)

---


# DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification

**DEQ-KAN** combines:
- **Deep Equilibrium Models (DEQs):** Implicit infinite-depth networks defined by fixed-point iteration (rather than explicitly stacking layers).
- **Kolmogorov–Arnold Networks (KANs):** A “dual” to MLPs, with learnable univariate functions on edges inspired by the Kolmogorov–Arnold (K–A) representation theorem.
- **CNN-based Feature Extraction:** Typically used up front to capture spatial patterns (e.g., in medical images).

This repository provides the **DEQ-KAN** approach, including code, usage instructions, and guidance for tasks such as:
- **Pneumonia detection** in chest X-ray images,
- **Multi-class brain tumor classification** in MRI scans,
- **Benign-vs.-malignant** histopathology image classification.

It achieves **strong robustness**, **high accuracy**, and **efficient memory usage**, thanks to the implicit modeling of an “infinite stack” of CNN+KAN layers.

---

## Table of Contents

1. [Introduction & Background](#introduction--background)  
2. [Methodology Overview](#methodology-overview)  
3. [Experiments & Key Results](#experiments--key-results)  
4. [Implementation & Code Link](#implementation--code-link)  
5. [Installation & Requirements](#installation--requirements)  
6. [Advice on Hyperparameter Tuning](#advice-on-hyperparameter-tuning)  
7. [Citation](#citation)  
8. [Contact](#contact)  
9. [Author's Note](#authors-note)

---

## Introduction & Background
**Kolmogorov–Arnold Networks (KANs)** place univariate basis expansions on each dimension-to-dimension “edge,” rather than having a single global activation on each node. They are mathematically grounded in the Kolmogorov–Arnold representation theorem and can be more expressive and interpretable than standard MLPs.

**Deep Equilibrium Models (DEQs)** eliminate explicit layers by formulating a single transformation and iterating it until convergence. This “infinite-depth” representation uses only the memory needed for one layer’s parameters.

By **merging DEQs with KANs**, **DEQ-KAN** repeatedly refines a hidden state in tandem with CNN-extracted features, enabling better classification results across imbalanced, multi-class, or small-image tasks in medical imaging.

---

## Methodology Overview
1. **CNN-Based Feature Extraction:** A typical CNN backbone processes raw images to produce meaningful feature vectors.  
2. **DEQ Iterations:** We define a transformation \(F(\mathbf{z})\) that includes both CNN features and KAN expansions, then iteratively solve \(\mathbf{z}^{\ast} = F(\mathbf{z}^{\ast})\).  
3. **KAN Blocks:** Each iteration uses univariate expansions \(\phi_{j,i}(z_i)\) for each edge \((i\rightarrow j)\), providing fine-grained modeling capacity.  
4. **Implicit Deep Network:** Instead of stacking many CNN+KAN layers, one implicit layer is iterated until convergence. Memory usage remains closer to a single layer, while effectively modeling infinite depth.

In our ablation studies, we show that both the **iterative** nature of DEQs and the **univariate expansions** of KAN together yield superior performance compared to feedforward baselines.

---

## Experiments & Key Results
We validate **DEQ-KAN** on:
- **Chest X-ray (Pneumonia Detection):** Achieved top accuracy and F1 scores despite class imbalance.
- **Brain Tumor (MRI) Classification:** Demonstrated superior ability to separate multiple tumor classes.
- **Histopathology (Benign vs. Malignant):** Accurately distinguishes subtle morphological features in small images.

Quantitative metrics (accuracy, precision, recall, specificity, ROC AUC) and confusion matrices consistently outperform traditional CNNs, Transformers, and other baselines.

---

## Implementation & Code Link
You can find the **complete implementation** (including the dataset class, CNN backbone, KAN blocks, DEQ routine, and a training script with warm-up, adaptive LR, robust init, and dropout) in our [codes folder](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes).  

The training script handles:
- **Gradual Warm-Up** of the learning rate,  
- **Adaptive LR scheduling** (step-based decay after warm-up),  
- **Xavier initialization** for CNN and KAN parameters,  
- **Dropout** in the KAN block for regularization,  
- **GPU or CPU** execution (with early stopping).

---

## Installation & Requirements
Below is an example setup referencing [pykan](https://github.com/KindXiaoming/pykan) conventions. Adapt as needed for **DEQ-KAN**:

**Requirements**
```bash
python==3.9.7
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.2.2
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml
```

**Efficiency Mode**
```bash
When you manually write the training loop and do not use the symbolic branch, call:
model.speed()


---

## Citation

If you use **DEQ-KAN** in your work, please cite this paper as follows:

```bibtex
@article{qezelbash2025DEQKAN,
  title={DEQ-KAN: Deep Equilibrium Kolmogorov-Arnold Networks for Robust Classification},
  author={Qezelbash-Chamak, Jaber},
  journal={Biomedical Signal Processing and Control},
  year={2025}
}

