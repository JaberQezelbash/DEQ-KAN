# weights_scheduler.py
import torch.nn as nn
import torch.optim as optim

def weights_init(module):
    """
    Apply Xavier initialization to Conv2d and Linear layers.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Gradually warm up learning rate from warmup_start_lr to base_lr over warmup_epochs.
    After warm-up, use the provided after_scheduler (e.g., StepLR) for further adjustments.
    """
    def __init__(self, optimizer, warmup_start_lr, base_lr, warmup_epochs, after_scheduler):
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if not self.finished:
            progress = min(self.last_epoch / self.warmup_epochs, 1.0)
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
            return [lr for _ in self.optimizer.param_groups]
        return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            if self.last_epoch < self.warmup_epochs:
                lr = self.get_lr()
                for param_group, lr_val in zip(self.optimizer.param_groups, lr):
                    param_group['lr'] = lr_val
            else:
                self.finished = True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr
                self.after_scheduler.step(epoch - self.warmup_epochs)

if __name__ == "__main__":
    # Quick test for weights initialization and scheduler
    import torch
    model = nn.Linear(10, 1)
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = GradualWarmupScheduler(optimizer, warmup_start_lr=1e-4, base_lr=1e-3,
                                       warmup_epochs=2,
                                       after_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5))
    print("Initial LR:", optimizer.param_groups[0]['lr'])
