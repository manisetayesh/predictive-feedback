import torch
class MultiLayerPerceptron(torch.nn.Module):
    """
    Simple multilayer perceptron model class with one hidden layer.
    """
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden=100,
        activation_type="sigmoid",
        bias=False,
    ):
        """
        Initializes a multilayer perceptron with a single hidden layer.

        Arguments:
        - num_inputs (int): number of input units (i.e., image size)
        - num_outputs (int): number of output units (i.e., number of
          classes)
        - num_hidden (int, optional): number of hidden units in the hidden layer
        - activation_type (str, optional): type of activation to use for the hidden
          layer ('sigmoid', 'tanh', 'relu' or 'linear')
        - bias (bool, optional): if True, each linear layer will have biases in
          addition to weights
        """

        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation_type = activation_type
        self.bias = bias

        # default weights (and biases, if applicable) initialization is used
        # see https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
        self.lin1 = torch.nn.Linear(num_inputs, num_hidden, bias=bias)
        self.lin2 = torch.nn.Linear(num_hidden, num_outputs, bias=bias)

        self._store_initial_weights_biases()

        self._set_activation()  # activation on the hidden layer
        self.softmax = torch.nn.Softmax(dim=1)  # activation on the output layer

    def _store_initial_weights_biases(self):
        """
        Stores a copy of the network's initial weights and biases.
        """
        self.init_lin1_weight = self.lin1.weight.data.clone()
        self.init_lin2_weight = self.lin2.weight.data.clone()
        if self.bias:
            self.init_lin1_bias = self.lin1.bias.data.clone()
            self.init_lin2_bias = self.lin2.bias.data.clone()

    def _set_activation(self):
        """
        Sets the activation function used for the hidden layer.
        """

        if self.activation_type.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid()  # maps to [0, 1]
        elif self.activation_type.lower() == "tanh":
            self.activation = torch.nn.Tanh()  # maps to [-1, 1]
        elif self.activation_type.lower() == "relu":
            self.activation = torch.nn.ReLU()  # maps to positive
        elif self.activation_type.lower() == "identity":
            self.activation = torch.nn.Identity()  # maps to same
        else:
            raise NotImplementedError(
                f"{self.activation_type} activation type not recognized. Only "
                "'sigmoid', 'relu' and 'identity' have been implemented so far."
            )

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets. This variable is not used
          here. However, it may be needed for other learning rules, to it is
          included as an argument here for compatibility.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.
        """

        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred

    def forward_backprop(self, X):
        """
        Identical to forward(). Should not be overwritten when creating new
        child classes to implement other learning rules, as this method is used
        to compare the gradients calculated with other learning rules to those
        calculated with backprop.
        """

        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred

    def list_parameters(self):
        """
        Returns a list of model names for a gradient dictionary.

        Returns:
        - params_list (list): List of parameter names.
        """

        params_list = list()

        for layer_str in ["lin1", "lin2"]:
            params_list.append(f"{layer_str}_weight")
            if self.bias:
                params_list.append(f"{layer_str}_bias")

        return params_list

    def gather_gradient_dict(self):
        """
        Gathers a gradient dictionary for the model's parameters. Raises a
        runtime error if any parameters have no gradients.

        Returns:
        - gradient_dict (dict): A dictionary of gradients for each parameter.
        """

        params_list = self.list_parameters()

        gradient_dict = dict()
        for param_name in params_list:
            layer_str, param_str = param_name.split("_")
            layer = getattr(self, layer_str)
            grad = getattr(layer, param_str).grad
            if grad is None:
                raise RuntimeError("No gradient was computed")
            gradient_dict[param_name] = grad.detach().clone().numpy()

        return gradient_dict


class BasicOptimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, weight_decay=0):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """

        for group in self.param_groups:
            for p in group["params"]:

                # only update parameters with gradients
                if p.grad is not None:

                    # apply weight decay to gradient, if applicable
                    if group["weight_decay"] != 0:
                        p.grad = p.grad.add(p, alpha=group["weight_decay"])

                    # apply gradient-based update
                    p.data.add_(p.grad, alpha=-group["lr"])
