### Hyperparameters & Implementation Details

Below is a brief summary of core hyperparameters and settings used in our experiments. Most were tuned manually for stable convergence, reduced overfitting, or improved runtime.

| **Hyperparameter**               | **Default** | **Explored**  | **Tuning Method**              | **Notes**                                                                                                                    |
|----------------------------------|------------:|--------------:|--------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Batch Size**                   | 16          | 8, 32         | Manual search                  | Kept moderate to accommodate CPU memory and mitigate over/underfitting.                                                     |
| **Learning Rate**                | 1e-3        | 1e-4,<br/>1e-2    | Lerning Rate Scheduler         | 1e-3 balanced convergence speed and stability. Too high => divergence; too low => slow training.                            |
| **DEQ Iterations**               | 10          | 5–15          | Manual search                  | Iterations ensure hidden-state convergence without excessive overhead.                                                      |
| **Relaxation Coefficient ($\alpha$)** | 0.5        | 0.1–1.0       | Manual search (step $=0.1$)               | $\alpha = 0.5$ gave stable convergence; smaller $\alpha$ requires more iterations.                                      |
| **Gaussian Kernels in KAN**    | 32          | 16, 64        | Manual search                  | Enough basis functions for flexible expansions; too many can overfit.                                                       |
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
- GPU Compatibility: On modern GPUs (e.g., RTX 3080), we observed ~4–10× faster training. DEQ-KAN remains memory-efficient since we differentiate only a single implicit module.
