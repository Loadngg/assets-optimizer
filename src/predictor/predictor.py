import math

import torch

from predictor import MeshExpertSystem


class Predictor:
    _CLASSES = {
        0: "High-Poly Sculpt / Raw Scan",
        1: "Animated Character / Creature",
        2: "Environment / Static Prop"
    }

    def __init__(self, model_weights_path=None):
        """
        Initialize the AI optimizer module.
        """
        self.model = MeshExpertSystem(input_features=2)
        self.model.eval()  # Set to inference mode

        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path))

    @staticmethod
    def _prepare_features(metadata: dict) -> torch.Tensor:
        """
        Transforms raw metadata into a feature vector for the neural network.
        Expects only 'polygons' and 'vertices'.
        """
        # Safe log calculation to handle large variations in mesh density
        poly_log = math.log10(metadata.get('polygons', 1) + 1)
        vert_log = math.log10(metadata.get('vertices', 1) + 1)

        # Build the vector matching the input_features=2 requirement
        feature_vector = [poly_log, vert_log]
        return torch.tensor([feature_vector], dtype=torch.float32)

    def analyze_model(self, metadata: dict) -> tuple:
        """
        Main entry point for external calls.
        """
        x = self._prepare_features(metadata)

        with torch.no_grad():  # Disable gradients for faster inference
            logits, criteria = self.model(x)

            # Get the index of the highest value for classification
            predicted_class_idx: int = int(torch.argmax(logits, dim=1).item())

            # Extract criteria values (index 0 for weight, index 1 for quality)
            weight_importance = criteria[0, 0].item()
            quality_importance = criteria[0, 1].item()

        predicted_class = self._CLASSES.get(predicted_class_idx, "Unknown")
        weight_importance = round(weight_importance, 3)
        quality_importance = round(quality_importance, 3)

        return predicted_class, weight_importance, quality_importance
