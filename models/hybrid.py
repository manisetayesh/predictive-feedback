import torch
import torch.nn as nn
from typing import Literal

from models.feedback import FAPerceptron


class HybridMLP(torch.nn.Module):
    def __init__(
        self,
        m1: nn.Module,
        m2: nn.Module,
        combination_method: Literal[
            "average",
            "weighted",
            "alternating",
            "gating",
            "concat",
            "chain",
            "stitch",
        ] = "average",
        **kwargs,
    ):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

        self.num_outputs = m1.num_outputs
        self.combination_method = combination_method
        self.alpha = 0.5
        self.step_count = 0

        if self.combination_method == "weighted":
            self.weight_param = nn.Parameter(
                torch.tensor(self.alpha, dtype=torch.float32)
            )
        elif self.combination_method == "stitch":
            first_m1 = list(self.m1.children())[0]
            rest_of_m2 = nn.Sequential(*list(self.m2.children())[1:])
            input_dim = first_m1.out_features
            expected_dim = list(self.m2.children())[1].in_features
            adapter = nn.Linear(input_dim, expected_dim)
            self.mlp = nn.Sequential(first_m1, adapter, *rest_of_m2)

        elif self.combination_method == "chain":
            self.mlp = type(self.m2)(
                num_inputs=self.m1.num_outputs,
                num_outputs=self.m2.num_outputs,
                num_hidden=self.m2.num_hidden,
                activation_type=self.m2.activation_type,
                bias=True,
            )
        else:
            NUM_HIDDEN = kwargs.get("num_hidden", self.m1.num_hidden)
            ACTIVATION = kwargs.get("num_hidden", self.m1.activation_type)
            self.mlp = FAPerceptron(
                num_inputs=self.num_outputs * 2,
                num_outputs=self.num_outputs,
                num_hidden=NUM_HIDDEN,
                activation_type=ACTIVATION,
                bias=True,
            )

    def forward(self, X, y=None):
        """
        Forward pass combining both algorithms.
        """

        m1_output = self.m1(X, y)
        m2_output = self.m2(X, y)
        method = self.combination_method

        if method == "chain":
            combined_output = self.mlp(m1_output, y)
        elif method == "stitch":
            combined_output = self.mlp(X.reshape(-1, self.m1.num_inputs))
        elif method == "average":
            combined_output = (m1_output + m2_output) / 2.0
        elif method== "weighted":
            combined_output = self._weighted_combination(m1_output, m2_output)
        elif method == "alternating":
            combined_output = self._alternating_combination(m1_output, m2_output)
        elif method == "gating":
            gate = self._concat_combination(m1_output, m2_output)
            combined_output = gate * m1_output + (1 - gate) * m2_output
        elif method == "concat":
            combined_output = self._concat_combination(m1_output, m2_output)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        return combined_output, m1_output, m2_output

    def _alternating_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        self.step_count += 1
        if self.step_count % 2 == 0:
            return m1_output
        else:
            return m2_output

    def _weighted_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        weight_m1 = torch.sigmoid(self.weight_param)
        return weight_m1 * m1_output + (1 - weight_m1) * m2_output

    def _concat_combination(
        self, m1_output: torch.Tensor, m2_output: torch.Tensor
    ) -> torch.Tensor:
        combined_outputs = torch.cat((m1_output, m2_output), dim=-1)
        return self.mlp(combined_outputs)

    def list_parameters(self):
        """
        Extended parameter list including hybrid components.

        Returns:
        - params_list (list): List of parameter names including hybrid components
        """
        params_list = []
        for name, _ in self.named_parameters():
            if not name.startswith("m1.") and not name.startswith("m2."):
                params_list.append(name)
        return params_list

    def gather_gradient_dict(self):
        """
        Extended gradient gathering including hybrid components.

        Returns:
        - gradient_dict (dict): Dictionary of gradients for all parameters
        """
        gradient_dict = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_dict[name] = param.grad.detach().clone().numpy()

        return gradient_dict
