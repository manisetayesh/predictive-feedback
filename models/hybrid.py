import torch
from mlp import MultiLayerPerceptron
from typing import Literal
class HybridMLP(MultiLayerPerceptron):
    def __init__(
        self,
        m1,
        m2,
        combination_method: Literal[
            "average", "weighted", "alternating"
        ] = "average",
        alpha: float = 0.5,
        **kwargs
    ):
        """
        Initialize hybrid MLP with two algorithms
        Arguments:
        - m1, m2: different algorithms
        - combination_method (str): how to combine algorithm outputs
        - alpha (float): weighting factor for weighted combination
        """
        super().__init__(**kwargs)
        self.m1 = m1
        self.m2 = m2
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
        m1_output = self.m1(X, y)
        m2_output = self.m2(X, y)
        if self.combination_method == "average":
            return self._average_combination(m1_output, m2_output)
        elif self.combination_method == "weighted":
            return self._weighted_combination(m1_output, m2_output)
        elif self.combination_method == "alternating":
            return self._alternating_combination(m1_output, m2_output)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _average_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        return (m1_output + m2_output) / 2.0

    def _weighted_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        return (1 - self.alpha) * m1_output + self.alpha * m2_output

    def _alternating_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        self.step_count += 1
        if self.step_count % 2 == 0:
            return m1_output
        else:
            return m2_output

    def list_parameters(self):
        """
        Extended parameter list including hybrid components.

        Returns:
        - params_list (list): List of parameter names including hybrid components
        """
        params_list = super().list_parameters()
        # Add sub-algorithm parameters
        for name, _ in self.m1.named_parameters():
            params_list.append(f"m1.{name}")
        for name, _ in self.m2.named_parameters():
            params_list.append(f"m2.{name}")

        return params_list

    def gather_gradient_dict(self):
        """
        Extended gradient gathering including hybrid components.

        Returns:
        - gradient_dict (dict): Dictionary of gradients for all parameters
        """
        gradient_dict = super().gather_gradient_dict()
        # Add gradients from each alg
        for name, param in self.m1.named_parameters():
            if param.grad is not None:
                gradient_dict[f"m1.{name}"] = (
                    param.grad.detach().clone().numpy()
                )
        for name, param in self.m2.named_parameters():
            if param.grad is not None:
                gradient_dict[f"m2.{name}"] = (
                    param.grad.detach().clone().numpy()
                )

        return gradient_dict
