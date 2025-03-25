# DEQ-KAN
DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification



<img width="600" alt="kan_plot" src="https://github.com/JaberQezelbash/DEQ-KAN/blob/main/assets/DEQ-KAN.svg">


This repository is the original DEQ-KAN concept as presented in this paper:

> **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**  
> *by Jaber Qezelbash-Chamak*.  
> [[Paper Link]](https://github.com/JaberQezelbash/DEQ-KAN)

---



## DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification

**DEQ-KAN** combines:
- **Deep Equilibrium Models (DEQs):** Implicit infinite-depth networks defined by fixed-point iteration (rather than explicitly stacking layers).
- **Kolmogorov–Arnold Networks (KANs):** A “dual” to MLPs, with learnable univariate functions on edges inspired by the Kolmogorov–Arnold (K–A) representation theorem.
- **CNN-based Feature Extraction:** Typically used up front to capture spatial patterns (e.g., in medical images).

This approach yields **strong robustness**, **high accuracy**, and **efficient memory usage**, thanks to the implicit modeling of an “infinite stack” of CNN+KAN layers. We show its utility in:
- **Pneumonia detection** in chest X-ray images,
- **Multi-class brain tumor classification** in MRI scans,
- **Benign-vs.-malignant** histopathology image classification.



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



## Introduction & Background

**Kolmogorov–Arnold Networks (KANs)** place univariate basis expansions on each dimension-to-dimension “edge,” rather than having a single global activation on each node. They are mathematically grounded in the Kolmogorov–Arnold representation theorem and can be more expressive and interpretable than standard MLPs.

**Deep Equilibrium Models (DEQs)** eliminate explicit layers by formulating a single transformation and iterating it until convergence. This “infinite-depth” representation uses only the memory needed for one layer’s parameters.

By **merging DEQs with KANs**, **DEQ-KAN** repeatedly refines a hidden state in tandem with CNN-extracted features, enabling better classification results across imbalanced, multi-class, or small-image tasks in medical imaging.



## Methodology Overview

1. **CNN-Based Feature Extraction**: A typical CNN backbone processes raw images to produce meaningful feature vectors.  
2. **DEQ Iterations**: We define a transformation \(F(\mathbf{z})\) that includes both CNN features and KAN expansions, then iteratively solve \(\mathbf{z}^{\ast} = F(\mathbf{z}^{\ast})\).  
3. **KAN Blocks**: Each iteration uses univariate expansions \(\phi_{j,i}(z_i)\) for each edge \((i\to j)\), providing fine-grained modeling capacity.  
4. **Implicit Deep Network**: Instead of stacking many CNN+KAN layers, one implicit layer is iterated until convergence. Memory usage remains closer to a single layer, while effectively modeling infinite depth.

Ablation studies demonstrate that the iterative DEQ mechanism and KAN’s univariate expansions jointly yield superior performance over standard feedforward networks.



## Experiments & Key Results

We validate **DEQ-KAN** on:

- **Chest X-ray (Pneumonia Detection)**: Achieved top accuracy and F1 scores despite class imbalance.  
- **Brain Tumor (MRI) Classification**: Demonstrated superior ability to separate multiple tumor classes.  
- **Histopathology (Benign vs. Malignant)**: Accurately distinguishes subtle morphological features in small images.

Our results consistently outperform baselines (CNNs, Transformers, etc.) across metrics like accuracy, precision, recall, specificity, and ROC AUC.



## Implementation & Code Link

The **complete implementation** (including dataset classes, CNN backbone, KAN blocks, DEQ routines, and training script with warm-up, adaptive LR, robust initialization, and dropout) is in the [codes folder](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes).  

### Hyperparameters & Implementation Details

Below is a brief summary of core hyperparameters and settings used in our experiments. Most were tuned manually for stable convergence, reduced overfitting, or improved runtime.

| **Hyperparameter**               | **Default** | **Explored**  | **Tuning Method**              | **Notes**                                                                                                                    |
|----------------------------------|------------:|--------------:|--------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Batch Size**                   | 16          | 8, 32         | Manual search                  | Kept moderate to accommodate CPU memory and mitigate over/underfitting.                                                     |
| **Learning Rate**                | 1e-3        | 1e-4, 1e-2    | Manual search                  | 1e-3 balanced convergence speed and stability. Too high => divergence; too low => slow training.                            |
| **DEQ Iterations**               | 10          | 5–15          | Manual search                  | Iterations ensure hidden-state convergence without excessive overhead.                                                      |
| **Relaxation Coefficient (\alpha)** | 0.5        | 0.1–1.0       | Manual search                  | \(\alpha = 0.5\) gave stable convergence; smaller \alpha might require more iterations.                                      |
| **# Gaussian Kernels in KAN**    | 32          | 16, 64        | Manual search                  | Enough basis functions for flexible expansions; too many can overfit.                                                       |
| **Epochs**                       | 50          | 30–70         | Early stopping                 | Past ~50 epochs, gains were minor; we used validation-based early stopping.                                                 |

Additional environment details for reproducibility:

| **Detail**           | **Setting/Choice**                          | **Notes**                                                                                          |
|----------------------|---------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Framework**        | PyTorch 2.5.0                               | Dynamic computation graphs and easy custom layer integration.                                       |
| **Language**         | Python 3.13.0                               | Offers robust library support and readability.                                                      |
| **Hardware**         | Intel Core i7 @3.6GHz, 24 GB RAM, CPU-only  | Ensures transparent run times; no GPU concurrency used for official results.                        |
| **OS**               | Windows 11                                  | Provided stable environment for reproducible CPU benchmarks.                                        |
| **Data Augmentation**| Flip, rotation up to 15°, normalization     | Minimal transformations to preserve subtle pathological details while reducing overfitting risk.     |
| **Optimizer**        | Adam (weight decay=1e-5)                    | Balanced speed and stability. Weight decay for additional regularization.                            |

### Computational Effort

- DEQ-KAN’s iterative fixed-point approach means multiple forward passes per sample. We cap iterations (e.g., ~10) and check for convergence.
- On a CPU-only environment, average epoch times ranged from 45–115 minutes, depending on dataset size (X-ray, MRI, or histopathology).  
- Empirically, convergence typically occurred in 7–12 iterations, adding modest overhead relative to large performance gains.
- **GPU Compatibility**: On modern GPUs (e.g., RTX 3080), we observed ~4–10× faster training. DEQ-KAN remains memory-efficient since we differentiate only a single implicit module.



## Installation & Requirements

Below is an example setup referencing [pykan](https://github.com/KindXiaoming/pykan). Adapt as needed for **DEQ-KAN**.

**Requirements** (example versions):
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
```





## Citation

If you use **DEQ-KAN** in your work, please cite this paper as follows:

```bibtex
@article{qezelbash2025DEQKAN,
  title={DEQ-KAN: Deep Equilibrium Kolmogorov-Arnold Networks for Robust Classification},
  author={Qezelbash-Chamak, Jaber},
  journal={Biomedical Signal Processing and Control},
  year={2025}
}
```

## Contact
For any questions related to DEQ-KAN, please contact:
[qezelbashc.jaber@ufl.edu](qezelbashc.jaber@ufl.edu)


## Author's Note
I appreciate your interest in DEQ-KAN. 
The aim is to provide a robust, memory-friendly, and interpretability-enhanced solution for challenging tasks like medical imaging classification. 
Feedback and collaboration inquiries are welcome!
