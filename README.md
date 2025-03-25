# DEQ-KAN
DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification




<img width="600" alt="kan_plot" src="https://github.com/JaberQezelbash/DEQ-KAN/blob/main/assets/DEQ-KAN.svg">


This repository extends the original Kolmogorov-Arnold Networks (KANs) concept to include our new method, **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks**, as presented in our paper:

> **DEQ-KAN: Deep Equilibrium Kolmogorov–Arnold Networks for Robust Classification**  
> *by Jaber Qezelbash-Chamak*.  
> [[Paper Link]](https://github.com/JaberQezelbash/DEQ-KAN) (Implementation Code)  

**DEQ-KAN** unifies [Deep Equilibrium Models (DEQs)](https://arxiv.org/abs/1909.01377) with [Kolmogorov-Arnold Networks(KANs)](https://arxiv.org/abs/2404.19756), providing an infinite-depth framework that repeatedly refines its feature representations. It has proven effective in challenging tasks like **medical image classification** (pneumonia detection in X-rays, brain tumor classification, and histopathology analysis). DEQ-KAN achieves strong robustness, high accuracy, and efficient memory usage by implicitly modeling a “stack” of infinitely many CNN+KAN layers.

---

## Table of Contents

- [Introduction & Background](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Introduction.md)
- [Methodology](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Methodology.md)
- [Experiments and Results](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Experiments.md)
- [Ablation Studies](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Ablation.md)
- [Conclusion and Future Work](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Conclusion.md)
- [Appendix: Configurations & Computational Details](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Appendix.md)
- [Implementation Code](#implementation-code)
  - [Dataset and Preprocessing](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/DataLoading.md)
  - [Model Architecture](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/ModelArchitecture.md)
  - [Training Script & LR Schedulers](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/TrainingScript.md)
  - [Weight Initialization & Other Utilities](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/LRScheduler.md)
- [Citation](#citation)
- [Contact](#contact)
- [Author's Note](#authors-note)

---

## Introduction & Background

In our paper, we motivate the design of **DEQ-KAN** by contrasting traditional multilayer perceptrons (MLPs) with Kolmogorov-Arnold Networks (KANs) and highlighting the benefits of implicit infinite-depth architectures (DEQs). For a full introduction and background, please refer to the detailed section:  
[Introduction & Background](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Introduction.md)

---

## Methodology

Our methodology integrates three core components:
- **Deep Equilibrium Models (DEQs):** Replace explicit stacking of layers with a fixed-point iterative process.
- **Kolmogorov-Arnold Networks (KANs):** Introduce learnable univariate basis expansions on edges for enhanced expressivity and interpretability.
- **CNN-based Feature Extraction:** Utilize convolutional layers to capture spatial cues before the DEQ-KAN iterative process.

For complete details including equations, convergence analysis, and model architecture, see:  
[Methodology](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Methodology.md)

---

## Experiments and Results

We validate **DEQ-KAN** on three medical image classification tasks:
- **Pneumonia Detection** in chest X-rays.
- **Brain Tumor Classification** in MRI scans.
- **Histopathology Analysis** (benign vs. malignant).

Comprehensive quantitative metrics and visualization (confusion matrices, ROC curves, t-SNE plots) are discussed in our paper. For the experimental setup and detailed results, visit:  
[Experiments and Results](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Experiments.md)

---

## Ablation Studies

To understand the contribution of each component, we performed ablation studies on our model. We compare variants with and without the DEQ iterations, KAN blocks, and CNN-based feature extractors. The results demonstrate that both the iterative DEQ mechanism and the univariate expansions of KAN are crucial for superior performance.  
Read more about these studies here:  
[Ablation Studies](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Ablation.md)

---

## Conclusion and Future Work

Our work demonstrates that DEQ-KAN achieves robust classification with high accuracy and efficiency. We discuss its implications for medical imaging and outline potential directions for further research, such as extending the method to additional modalities and optimizing for real-time deployment.  
For a complete conclusion and future outlook, see:  
[Conclusion and Future Work](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Conclusion.md)

---

## Appendix: Configurations & Computational Details

The appendix provides full details on hyperparameter settings, training configurations, and computational cost analyses. This section is essential for reproducing our results and understanding the practical considerations of our approach.  
Access the appendix here:  
[Appendix: Configurations & Computational Details](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/Appendix.md)

---

## Implementation Code

The full implementation is available in the repository and is divided into several parts for clarity. You can browse each component via the following links:

- **Dataset and Preprocessing:**  
  [DataLoading.md](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/DataLoading.md)
  
- **Model Architecture:**  
  [ModelArchitecture.md](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/ModelArchitecture.md)
  
- **Training Script & Learning Rate Schedulers:**  
  [TrainingScript.md](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/TrainingScript.md)
  
- **Weight Initialization & Utility Functions:**  
  [LRScheduler.md](https://github.com/JaberQezelbash/DEQ-KAN/blob/main/codes/LRScheduler.md)

The provided code includes advanced features such as gradual warm-up, adaptive learning rate scheduling, robust Xavier initialization, and dropout regularization in the KAN block.

---

## Citation

If you use **DEQ-KAN** in your work, please cite our paper as follows:

```bibtex
@article{qezelbash2024DEQKAN,
  title={DEQ-KAN: Deep Equilibrium Kolmogorov-Arnold Networks for Robust Classification},
  author={Qezelbash-Chamak, Jaber and ...},
  journal={Biomedical Signal Processing and Control},
  year={2024}
}

