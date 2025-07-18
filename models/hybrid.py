import torch
from mlp import MultiLayerPerceptron
from typing import Literal
class HybridMLP(MultiLayerPerceptron):
    def __init__(
        self,
        predictive_encoder,
        weight_mirroring,
        combination_method: Literal[
            "average", "weighted", "alternating"
        ] = "average",
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize hybrid MLP with predictive encoding and weight mirroring algorithms.

        Arguments:
        - predictive_encoder: Module implementing predictive encoding
        - weight_mirroring: Module implementing weight mirroring
        - combination_method (str): how to combine algorithm outputs
        - alpha (float): weighting factor for weighted combination
        """
        super().__init__(**kwargs)
        self.predictive_encoder = predictive_encoder
        self.weight_mirroring = weight_mirroring
        self.combination_method = combination_method
        self.alpha = alpha
        self.step_count = 0

    def forward(self, X, y=None):
        """
        Forward pass combining both algorithms.

        Arguments:
        - X (torch.Tensor): Batch of input images
        - y (torch.Tensor, optional): Batch of targets
        Returns:
        - y_pred (torch.Tensor): Combined predicted targets
        """
        pred_output = self.predictive_encoder(X, y)
        mirror_output = self.weight_mirroring(X, y)
        if self.combination_method == "average":
            return self._average_combination(pred_output, mirror_output)
        elif self.combination_method == "weighted":
            return self._weighted_combination(pred_output, mirror_output)
        elif self.combination_method == "alternating":
            return self._alternating_combination(pred_output, mirror_output)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _average_combination(
        self, pred_output: torch.Tensor, mirror_output: torch.Tensor
    ) -> torch.Tensor:
        return (pred_output + mirror_output) / 2.0

    def _weighted_combination(
        self, pred_output: torch.Tensor, mirror_output: torch.Tensor
    ) -> torch.Tensor:
        return (1 - self.alpha) * pred_output + self.alpha * mirror_output

    def _alternating_combination(
        self, pred_output: torch.Tensor, mirror_output: torch.Tensor
    ) -> torch.Tensor:
        self.step_count += 1
        if self.step_count % 2 == 0:
            return pred_output
        else:
            return mirror_output

    def list_parameters(self):
        """
        Extended parameter list including hybrid components.

        Returns:
        - params_list (list): List of parameter names including hybrid components
        """
        params_list = super().list_parameters()
        # Add sub-algorithm parameters
        for name, _ in self.predictive_encoder.named_parameters():
            params_list.append(f"predictive_encoder.{name}")
        for name, _ in self.weight_mirroring.named_parameters():
            params_list.append(f"weight_mirroring.{name}")

        return params_list

    def gather_gradient_dict(self):
        """
        Extended gradient gathering including hybrid components.

        Returns:
        - gradient_dict (dict): Dictionary of gradients for all parameters
        """
        gradient_dict = super().gather_gradient_dict()
        # Add gradients from each alg
        for name, param in self.predictive_encoder.named_parameters():
            if param.grad is not None:
                gradient_dict[f"predictive_encoder.{name}"] = (
                    param.grad.detach().clone().numpy()
                )
        for name, param in self.weight_mirroring.named_parameters():
            if param.grad is not None:
                gradient_dict[f"weight_mirroring.{name}"] = (
                    param.grad.detach().clone().numpy()
                )

        return gradient_dict
