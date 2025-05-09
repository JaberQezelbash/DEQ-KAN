# DEQ-KAN
**Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**



<img width="800" alt="kan_plot" src="https://github.com/JaberQezelbash/DEQ-KAN/blob/main/assets/DEQ-KAN.svg">


This repository is the original DEQ-KAN concept as presented in this paper:

> **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**  
> *by Jaber Qezelbash-Chamak* 
> [[Paper Link]](https://github.com/JaberQezelbash/DEQ-KAN)

---



## Motivation
Medical image classification is a critical yet challenging task, where even minor misclassifications can have serious clinical implications. Traditional deep learning models often require stacking numerous layers to capture complex patterns, leading to high memory usage and potential overfitting, especially in settings with limited or imbalanced data. DEQ-KAN is motivated by the need to overcome these hurdles by unifying [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) (DEQs)—which enable implicit, infinite-depth representations through iterative fixed-point convergence—with [Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs) that introduce flexible, learnable univariate activations on each network edge. This innovative synergy not only improves accuracy and robustness in tasks such as pneumonia detection, brain tumor classification, and histopathology analysis, but also enhances interpretability and computational efficiency, paving the way for more reliable and scalable medical imaging solutions.


## Table of Contents

1. [Introduction](#introduction)  
2. [Methodology Overview](#methodology-overview)  
3. [Experiments](#experiments)  
4. [Implementation & Codes](#implementation--codes)  
5. [Installation & Requirements](#installation--requirements)  
6. [Configurations](#configurations)  
7. [Citation](#citation)  
8. [Contact](#contact)  
9. [Author's Note](#authors-note)



## Introduction

**KANs** place univariate basis expansions on each dimension-to-dimension “edge,” rather than having a single global activation on each node. They are mathematically grounded in the Kolmogorov–Arnold representation theorem and can be more expressive and interpretable than standard MLPs.
**DEQs** eliminate explicit layers by formulating a single transformation and iterating it until convergence. This “infinite-depth” representation uses only the memory needed for one layer’s parameters.
By merging DEQs with KANs, **DEQ-KAN** repeatedly refines a hidden state in tandem with CNN-extracted features, enabling better classification results across imbalanced, multi-class, or small-image tasks in medical imaging.

DEQ-KAN combines:
- DEQs: Implicit infinite-depth networks defined by fixed-point iteration (rather than explicitly stacking layers).
- KANs: A “dual” to MLPs, with learnable univariate functions on edges inspired by the Kolmogorov–Arnold (K–A) representation theorem.
- CNN-based Feature Extraction: Used up front to capture spatial patterns (e.g., in medical images).



## Methodology Overview

1. **CNN-Based Feature Extraction**: A typical CNN backbone processes raw images to produce meaningful feature vectors.  
2. **DEQ Iterations**: We define a transformation $F(\mathbf{z})$ that includes both CNN features and KAN expansions, then iteratively solve $\mathbf{z}^{\ast} = F(\mathbf{z}^{\ast})$.  
3. **KAN Blocks**: Each iteration uses univariate expansions $\phi_{j,i}(z_i)$ for each edge $(i\to j)$, providing fine-grained modeling capacity.  
4. **Implicit Deep Network**: Instead of stacking many CNN+KAN layers, one implicit layer is iterated until convergence. Memory usage remains closer to a single layer, while effectively modeling infinite depth.

Ablation studies demonstrate that the iterative DEQ mechanism and KAN’s univariate expansions jointly yield superior performance over standard feedforward networks.



## Experiments

We validate DEQ-KAN on:

- Chest X-ray (Pneumonia Detection): Achieved top accuracy and F1 scores despite class imbalance.  
- Brain Tumor (MRI) Classification: Demonstrated superior ability to separate multiple tumor classes.  
- Histopathology (Benign vs. Malignant): Accurately distinguishes subtle morphological features in small images.

Our results consistently outperform baselines (CNNs, Transformers, etc.) across metrics like accuracy, precision, recall, specificity, and ROC AUC.



## Implementation & Codes

The complete implementation (including dataset classes, CNN backbone, KAN blocks, DEQ routines, training script with warm-up, adaptive LR, robust initialization, and dropout) is available in the [codes folder](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes).  

### Configurations 

For detailed configurations, hyperparameters, and computational efforts, please refer to [Configurations](assets/configurations.md).



## Installation & Requirements

Below is an example setup referencing [DEQ-KAN](https://github.com/JaberQezelbash/DEQ-KAN/).

**Requirements** (example versions):
```bash
python==3.13.0
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
setuptools==65.5.0
sympy==1.11.1
torch==2.5.0
tqdm==4.66.2
pandas==2.0.1
seaborn
pyyaml

```



<!-- 
## Citation

If you use DEQ-KAN in your work, please cite this paper as follows:

```bibtex
@article{qezelbash2025DEQKAN,
  title={DEQ-KAN: Deep Equilibrium Kolmogorov-Arnold Networks for Robust Classification},
  author={Qezelbash-Chamak, Jaber},
  journal={Biomedical Signal Processing and Control},
  year={2025}
}
```
-->


## Contact
For any questions related to DEQ-KAN, you may contact:
[qezelbashc.jaber@ufl.edu](qezelbashc.jaber@ufl.edu)


## Author's Note
I appreciate your interest in DEQ-KAN. 
The aim is to provide a robust, memory-friendly, and interpretability-enhanced solution for challenging tasks like medical imaging classification. 
Feedback and collaboration inquiries are welcome!
