# DEQ-KAN
DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification




<img width="600" alt="kan_plot" src="https://github.com/JaberQezelbash/DEQ-KAN/blob/main/assets/DEQ-KAN.svg">


This repository extends the original Kolmogorov-Arnold Networks (KANs) concept to include our new method, **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks**, as presented in our paper:

> **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**  
> *by Jaber Qezelbash-Chamak*.  
> [[Paper Link]](https://github.com/JaberQezelbash/DEQ-KAN) (Implementation Code)  

**DEQ-KAN** unifies Deep Equilibrium Models (DEQs) with Kolmogorov-Arnold expansions, providing an infinite-depth framework that repeatedly refines its feature representations. It has proven effective in challenging tasks like **medical image classification** (pneumonia detection in X-rays, brain tumor classification, and histopathology analysis). DEQ-KAN achieves strong robustness, high accuracy, and efficient memory usage by implicitly modeling a “stack” of infinitely many CNN+KAN layers.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [DEQ-KAN Overview](#deq-kan-overview)
4. [Usage and Examples](#usage-and-examples)
5. [Computation Requirements](#computation-requirements)
6. [Documentation](#documentation)
7. [Tutorials](#tutorials)
8. [Hyperparameter Tuning Advice](#hyperparameter-tuning-advice)
9. [Citations](#citations)
10. [Contact](#contact)
11. [Author's Note](#authors-note)

---

## Introduction

Kolmogorov-Arnold Networks (KANs) were introduced as a “dual” to classic MLPs: instead of having activation functions on **nodes**, KANs have **activation functions on edges**. This structure is motivated by the Kolmogorov–Arnold (K–A) representation theorem, which states that any continuous multivariate function can be decomposed into sums of univariate functions of linear combinations of the inputs.

**Deep Equilibrium Kolmogorov–Arnold Networks (DEQ-KAN)** build upon this foundation by integrating:

- **Deep Equilibrium Models (DEQs)**: Implicit infinite-depth architectures that iteratively solve for a fixed point instead of stacking many explicit layers.
- **KAN expansions**: Learnable univariate functions on each edge, which capture subtleties and improve expressivity.
- **CNN-based Feature Extraction**: Typically used to capture spatial or structural cues from raw image data.

This hybrid approach has shown **state-of-the-art performance** in diverse medical imaging tasks, including:
- **Pneumonia Detection** using chest X-ray images,
- **Multi-class Brain Tumor Classification** from MRI scans,
- **Benign-vs.-Malignant Histopathology** image classification.

---

## Installation

You can install this package (which includes both KAN and the newly integrated DEQ-KAN code) via PyPI or directly from GitHub.  

**Pre-requisites:**
```
Python 3.9.7 or higher
pip
```

### For Developers
```bash
git clone https://github.com/KindXiaoming/pykan.git
cd pykan
pip install -e .
```

### Installation via GitHub
```bash
pip install git+https://github.com/KindXiaoming/pykan.git
```

### Installation via PyPI
```bash
pip install pykan
```

Be sure you also have these packages installed:
```python
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

### Optional: Conda Environment Setup
```bash
conda create --name pykan-env python=3.9.7
conda activate pykan-env
pip install git+https://github.com/KindXiaoming/pykan.git
# or
pip install pykan
```

---

## DEQ-KAN Overview

In **DEQ-KAN**, we embed **KAN** blocks (univariate expansions on edges) inside a **Deep Equilibrium** framework:

- **Deep Equilibrium (DEQ) Core**  
  Instead of explicitly stacking CNN or KAN layers, we define a single iterative transformation that refines a hidden state \(\mathbf{z}\) until it converges to a **fixed point**.

- **Kolmogorov-Arnold Networks (KAN)**  
  Each transformation includes univariate basis expansions \(\phi_{j,i}(z)\) for each dimension-to-dimension mapping, making the model more expressive and interpretable than global activations alone.

- **CNN-based Feature Extraction**  
  We typically use a CNN as a front-end to capture spatial features (especially beneficial for image data), then feed these features into the iterative DEQ-KAN pipeline.

This implicit architecture effectively emulates an *infinite-depth* network with relatively **low memory overhead**.

**Highlighted Advantages**:
1. **High Accuracy**  
   Achieved top metrics across multiple medical classification tasks.
2. **Robustness to Class Imbalance**  
   Iterative refinement makes it less likely to overlook minority classes.
3. **Memory Efficiency**  
   Only the final equilibrium state needs to be retained.
4. **Interpretability**  
   Univariate expansions clarify how each dimension influences predictions.

---

## Usage and Examples

Below is a conceptual snippet to illustrate how you might use DEQ-KAN for an image classification task. A more complete example can be found in [tutorials](#tutorials).

```python
import torch
from pykan import KAN   # base KAN classes

# Suppose you also have a CNN-based feature extractor
class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Your usual CNN layers here...
    def forward(self, x):
        # Return feature vector
        return features

class DEQKANModel(torch.nn.Module):
    def __init__(self, cnn_channels, hidden_dim):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        # Projection to hidden_dim
        self.proj = torch.nn.Linear(cnn_channels, hidden_dim)
        # KAN block
        self.kan_block = KAN(width=[hidden_dim, hidden_dim],
                             grid=32,
                             k=3)
        # Classifier
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        
        # Additional DEQ settings
        self.max_iter = 10
        self.alpha = 0.5
    
    def forward(self, x):
        # Extract CNN features
        f = self.cnn(x)
        z_init = torch.zeros(x.size(0), f.size(1), device=x.device)
        
        # DEQ Iterations
        z = z_init
        for _ in range(self.max_iter):
            # combine current z with CNN features
            z_next = z + self.alpha * (self.deq_transform(f, z) - z)
            # check for convergence or limit max_iter
            z = z_next
        return self.classifier(z)
    
    def deq_transform(self, f, z):
        # Project CNN features, add current hidden state, then pass through KAN
        combined = z + self.proj(f)
        return combined + self.kan_block(combined)
```

You can then train `DEQKANModel` using PyTorch’s usual training loop, or incorporate speed-ups using GPU parallelization.  

**Efficiency Mode**:  
If you don’t use the symbolic branch and want faster training:
```python
my_model.kan_block.speed()
```
before your main training loop.

---

## Computation Requirements

- Examples in the [tutorials](#tutorials) typically run on a single CPU in under 10 minutes.  
- **DEQ-KAN** tasks with large medical image datasets (e.g., big histopathology sets) can take several hours on a CPU-only environment.  
- For large-scale tasks, using a **GPU** is highly recommended.

---

## Documentation

The main KAN documentation is available [here](https://kindxiaoming.github.io/pykan/).  
- You’ll find references to classes like `KAN`, utility functions for plotting, and more.  
- For **DEQ-KAN**-specific functionalities, see the [Usage](#usage-and-examples) snippet above or integrated notebooks in the [tutorials](#tutorials).

---

## Tutorials

1. **Quickstart**  
   [hellokan.ipynb](./hellokan.ipynb) – Basic introduction to KAN usage.

2. **Extended Notebooks**  
   Additional notebooks for PDE examples, function fitting, and interpretability can be found in the [tutorials](tutorials) directory.

3. **Medical Image Classification with DEQ-KAN**  
   - See the `medical_image_deq_kan.ipynb` notebook (if provided) for a direct demonstration of how to set up pneumonia detection, brain tumor classification, or histopathology classification using the DEQ-KAN approach described in our paper.  

---

## Hyperparameter Tuning Advice

> *Much of this advice remains from the original KAN approach but also applies to DEQ-KAN, particularly for adjusting the univariate expansions.*

1. **Start Small**  
   For KAN shapes, smaller width and grid sizes often suffice, especially if your dataset is not massive. For DEQ-KAN, you can start with ~8–10 DEQ iterations and moderate kernel counts in the KAN layer.

2. **Iterate on Grid Extension**  
   If underfitting, consider raising the `grid` parameter in KAN. Check convergence in the DEQ loop by adjusting `max_iter` or `alpha`.

3. **Regularization**  
   If interpretability is essential, use `lamb` to sparsify or prune unneeded edges in KAN expansions. For medical tasks, controlling complexity can help reduce overfitting on subtle patterns.

4. **Class Imbalance**  
   The iterative nature of DEQ-KAN is already robust to minority classes, but standard techniques (e.g., class weighting or oversampling) can further enhance results.

5. **Check Underfitting/Overfitting**  
   - If large training/test loss gaps arise, consider more data or a smaller `grid`.  
   - If stuck in underfitting, enlarge the width or increase the CNN depth.  

For more details, see the “Methodology” and “Ablation Studies” sections of our [DEQ-KAN paper](https://github.com/JaberQezelbash/DEQ-KAN).

---

## Citation

If you use **DEQ-KAN** or build upon the DEQ-KAN approach, please cite:
```bibtex
@article{qezelbash2024DEQKAN,
  title={DEQ-KAN: Deep Equilibrium Kolmogorov-Arnold Networks for Robust Classification},
  author={Qezelbash-Chamak, Jaber and ...},
  journal={Biomedical Signal Processing and Control},
  year={2024}
}
```

---

## Contact

- For **DEQ-KAN**-specific questions or medical imaging implementations, please contact [qezelbashc@jaber@ufl.edu](mailto:qezelbashc@jaber@ufl.edu).

---

## Author's Note

We appreciate your interest in Kolmogorov-Arnold Networks. The original KAN code focuses on smaller-scale scientific tasks, but with the **DEQ-KAN** extension, we hope to broaden usability for high-stakes applications like medical diagnostics. Efficiency optimizations and advanced usage patterns are still evolving, so we welcome feedback, critiques, and collaboration ideas.

> *“Practice is the only criterion for testing understanding (实践是检验真理的唯一标准).”*

Feel free to open an issue or submit a pull request if you encounter any problems or have suggestions for making DEQ-KAN more efficient or applicable to new domains.

---

*Happy Researching & Coding with DEQ-KAN and KAN!*
