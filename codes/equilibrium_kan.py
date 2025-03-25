# equilibrium_kan.py
import torch
import torch.nn as nn
from cnn_backbone import DeepCNNBackbone
from kan_components import KANBlock

class EquilibriumKANClassifier(nn.Module):
    """
    DEQ-KAN classifier:
      F(x, z) = z + CNNProj(Backbone(x)) + KAN(z + CNNProj(Backbone(x)))
    Iterative update:
      z_{k+1} = z_k + alpha * [F(x, z_k) - z_k]
    """
    def __init__(self, num_classes=1, hidden_dim=128, num_knots=10,
                 max_iter=25, tol=1e-3, alpha=1.0, dropout_p=0.2):
        super(EquilibriumKANClassifier, self).__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

        # CNN Backbone
        self.backbone = DeepCNNBackbone(input_c=3, base_ch=32)
        # Linear projection (assumes backbone output dimension is 128)
        self.cnn_proj = nn.Linear(128, hidden_dim)
        # KAN Block
        self.kan_block = KANBlock(hidden_dim, num_knots=num_knots, dropout_p=dropout_p)
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward_function(self, x, z):
        feat = self.backbone(x)
        feat_proj = self.cnn_proj(feat)
        combined = z + feat_proj
        kan_out = self.kan_block(combined)
        return z + feat_proj + kan_out

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        # Initialize hidden state
        z = torch.zeros(batch_size, self.kan_block.kan1.dim_in, device=device)
        for _ in range(self.max_iter):
            Fz = self.forward_function(x, z)
            z_next = z + self.alpha * (Fz - z)
            diff = (z_next - z).norm(p=2, dim=1).mean().item()
            z = z_next
            if diff < self.tol:
                break
        logits = self.classifier(z).squeeze(-1)
        return logits

if __name__ == "__main__":
    # Quick test for EquilibriumKANClassifier
    model = EquilibriumKANClassifier(num_classes=1, hidden_dim=128,
                                     num_knots=10, max_iter=25, tol=1e-3,
                                     alpha=0.1, dropout_p=0.2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Classifier output shape:", out.shape)
