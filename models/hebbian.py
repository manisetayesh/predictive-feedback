import torch
from .mlp import MultiLayerPerceptron
class HebbianFunction(torch.autograd.Function):
    """
    Gradient computing function class for Hebbian learning.
    """

    @staticmethod
    def forward(context, input, weight, bias=None, nonlinearity=None, target=None):
        """
        Forward pass method for the layer. Computes the output of the layer and
        stores variables needed for the backward pass.

        Arguments:
        - context (torch context): context in which variables can be stored for
          the backward pass.
        - input (torch tensor): input to the layer.
        - weight (torch tensor): layer weights.
        - bias (torch tensor, optional): layer biases.
        - nonlinearity (torch functional, optional): nonlinearity for the layer.
        - target (torch tensor, optional): layer target, if applicable.

        Returns:
        - output (torch tensor): layer output.
        """

        # compute the output for the layer (linear layer with non-linearity)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        if nonlinearity is not None:
            output = nonlinearity(output)

        # calculate the output to use for the backward pass
        output_for_update = output if target is None else target

        # store variables in the context for the backward pass
        context.save_for_backward(input, weight, bias, output_for_update)

        return output

    @staticmethod
    def backward(context, grad_output=None):
        """
        Backward pass method for the layer. Computes and returns the gradients for
        all variables passed to forward (returning None if not applicable).

        Arguments:
        - context (torch context): context in which variables can be stored for
          the backward pass.
        - input (torch tensor): input to the layer.
        - weight (torch tensor): layer weights.
        - bias (torch tensor, optional): layer biases.
        - nonlinearity (torch functional, optional): nonlinearity for the layer.
        - target (torch tensor, optional): layer target, if applicable.

        Returns:
        - grad_input (None): gradients for the input (None, since gradients are not
          backpropagated in Hebbian learning).
        - grad_weight (torch tensor): gradients for the weights.
        - grad_bias (torch tensor or None): gradients for the biases, if they aren't
          None.
        - grad_nonlinearity (None): gradients for the nonlinearity (None, since
          gradients do not apply to the non-linearities).
        - grad_target (None): gradients for the targets (None, since
          gradients do not apply to the targets).
        """

        input, weight, bias, output_for_update = context.saved_tensors
        grad_input = None
        grad_weight = None
        grad_bias = None
        grad_nonlinearity = None
        grad_target = None

        input_needs_grad = context.needs_input_grad[0]
        if input_needs_grad:
            pass

        weight_needs_grad = context.needs_input_grad[1]
        if weight_needs_grad:
            grad_weight = output_for_update.t().mm(input)
            grad_weight = grad_weight / len(input)  # average across batch

            # center around 0
            grad_weight = grad_weight - grad_weight.mean(axis=0)  # center around 0

            ## or apply Oja's rule (not compatible with clamping outputs to the targets!)
            # oja_subtract = output_for_update.pow(2).mm(grad_weight).mean(axis=0)
            # grad_weight = grad_weight - oja_subtract

            # take the negative, as the gradient will be subtracted
            grad_weight = -grad_weight

        if bias is not None:
            bias_needs_grad = context.needs_input_grad[2]
            if bias_needs_grad:
                grad_bias = output_for_update.mean(axis=0)  # average across batch

                # center around 0
                grad_bias = grad_bias - grad_bias.mean()

                ## or apply an adaptation of Oja's rule for biases
                ## (not compatible with clamping outputs to the targets!)
                # oja_subtract = (output_for_update.pow(2) * bias).mean(axis=0)
                # grad_bias = grad_bias - oja_subtract

                # take the negative, as the gradient will be subtracted
                grad_bias = -grad_bias

        return grad_input, grad_weight, grad_bias, grad_nonlinearity, grad_target


class HebbianMultiLayerPerceptron(MultiLayerPerceptron):
    """
    Hebbian multilayer perceptron with one hidden layer.
    """

    def __init__(self, clamp_output=True, **kwargs):
        """
        Initializes a Hebbian multilayer perceptron object

        Arguments:
        - clamp_output (bool, optional): if True, outputs are clamped to targets,
          if available, when computing weight updates.
        """

        self.clamp_output = clamp_output
        super().__init__(**kwargs)

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets, stored for the backward
          pass to compute the gradients for the last layer.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.
        """

        h = HebbianFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.lin1.bias,
            self.activation,
        )

        # if targets are provided, they can be used instead of the last layer's
        # output to train the last layer.
        if y is None or not self.clamp_output:
            targets = None
        else:
            targets = torch.nn.functional.one_hot(
                y, num_classes=self.num_outputs
            ).float()

        y_pred = HebbianFunction.apply(
            h, self.lin2.weight, self.lin2.bias, self.softmax, targets
        )

        return y_pred
