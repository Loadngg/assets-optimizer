import torch.nn as nn


class MeshExpertSystem(nn.Module):
    def __init__(self, input_features=2):
        super().__init__()

        # Shared layers: extract common patterns from geometry metadata
        self.shared_layers = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Classification head (3 classes: High-Poly, Animated, Static)
        self.classifier_head = nn.Sequential(
            nn.Linear(16, 3)
        )

        # Criteria head: outputs 2 values (weight_importance, quality_importance)
        # Softmax ensures the sum of these two values is exactly 1.0
        self.criteria_head = nn.Sequential(
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        shared_features = self.shared_layers(x)

        logits = self.classifier_head(shared_features)
        criteria = self.criteria_head(shared_features)

        return logits, criteria
