
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from .mlp import MultiLayerPerceptron

def train_model_mirror(MLP, train_loader, valid_loader, optimizer, num_epochs=5, loss_type="NLL", mirror=False, m_lr=0.01, noise_amplitude=1):
  """
  Train a model for several epochs. Updated to work with FA and WM

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - num_epochs (int, optional): Number of epochs to train model.

  Returns:
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  """

  results_dict = {
      "avg_train_losses": list(),
      "avg_valid_losses": list(),
      "avg_train_accuracies": list(),
      "avg_valid_accuracies": list(),
  }

  for e in tqdm(range(num_epochs)):
    no_train = True if e == 0 else False # to get a baseline
    latest_epoch_results_dict = train_epoch_mirror(
        MLP, train_loader, valid_loader, optimizer=optimizer, loss_type=loss_type, no_train=no_train, mirror=mirror, m_lr=m_lr, noise_amplitude=noise_amplitude
        )

    for key, result in latest_epoch_results_dict.items():
      if key in results_dict.keys() and isinstance(results_dict[key], list):
        results_dict[key].append(latest_epoch_results_dict[key])
      else:
        results_dict[key] = result # copy latest

  return results_dict


def train_epoch_mirror(MLP, train_loader, valid_loader, optimizer, loss_type, no_train=False, mirror=False, m_lr=0.01, noise_amplitude=1):
  """
  Train a model for one epoch.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - no_train (bool, optional): If True, the model is not trained for the
    current epoch. Allows a baseline (chance) performance to be computed in the
    first epoch before training starts.

  Returns:
  - epoch_results_dict (dict): Dictionary storing epoch results on training
    and validation data.
  """

  if loss_type == "NLL":
    criterion = torch.nn.NLLLoss()
    do_log = True
  #Better loss type for FA
  elif loss_type == "CrossEntropy":
    criterion = torch.nn.CrossEntropyLoss()
    do_log = False
  else:
    raise NotImplementedError(
          f"{loss_type} loss type not recognized."
          )

  epoch_results_dict = dict()
  for dataset in ["train", "valid"]:
    for sub_str in ["correct_by_class", "seen_by_class"]:
      epoch_results_dict[f"{dataset}_{sub_str}"] = {
          i:0 for i in range(MLP.num_outputs)
          }

  MLP.train()
  train_losses, train_acc = list(), list()
  for X, y in train_loader:
    y_pred = MLP(X, y=y)
    #loss = criterion(torch.log(y_pred), y)
    if do_log:
      loss = criterion(torch.log(y_pred), y)
    else:
      loss = criterion(y_pred, y)
    
    acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
    train_losses.append(loss.item() * len(y))
    train_acc.append(acc.item() * len(y))
    update_results_by_class_in_place(
        y, y_pred.detach(), epoch_results_dict, dataset="train",
        num_classes=MLP.num_outputs
        )
    optimizer.zero_grad()
    if not no_train:
      loss.backward(retain_graph=True)
      optimizer.step()

    if mirror:
      MLP.mirror(X, m_lr, noise_amplitude)

  num_items = len(train_loader.dataset)
  epoch_results_dict["avg_train_losses"] = np.sum(train_losses) / num_items
  epoch_results_dict["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100

  MLP.eval()
  valid_losses, valid_acc = list(), list()
  with torch.no_grad():
    for X, y in valid_loader:
      y_pred = MLP(X)
      if do_log:
        loss = criterion(torch.log(y_pred), y)
      else:
        loss = criterion(y_pred, y)
      acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
      valid_losses.append(loss.item() * len(y))
      valid_acc.append(acc.item() * len(y))
      update_results_by_class_in_place(
          y, y_pred.detach(), epoch_results_dict, dataset="valid"
          )

  num_items = len(valid_loader.dataset)
  epoch_results_dict["avg_valid_losses"] = np.sum(valid_losses) / num_items
  epoch_results_dict["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

  return epoch_results_dict

class LinearFAFunction(torch.autograd.Function):

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa.to(grad_output.device))
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        #return grad_input, grad_weight, grad_bias, grad_nonlinearity, grad_target
        return grad_input, grad_weight, grad_bias, grad_weight_fa, None

class FAPerceptron(MultiLayerPerceptron):

  def __init__(self, clamp_output=True, **kwargs):

    super().__init__(**kwargs)
    #Create feedback weights
    self.lin1.weight_fa = Variable(torch.FloatTensor(self.num_hidden, self.num_inputs), requires_grad=False)
    self.lin2.weight_fa = Variable(torch.FloatTensor(self.num_outputs, self.num_hidden), requires_grad=False)
    #Initiate them randomly. In standard FA these are constant and never changed
    torch.nn.init.kaiming_uniform_(self.lin1.weight_fa)
    torch.nn.init.kaiming_uniform_(self.lin2.weight_fa)


  def forward(self, X, y=None):

    h = LinearFAFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1.weight_fa,
        self.lin1.bias
    )

    y_pred = LinearFAFunction.apply(
        h,
        self.lin2.weight,
        self.lin2.weight_fa,
        self.lin2.bias
    )

    return y_pred
  
class LinearWMFunction(torch.autograd.Function):

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        #output saved for Hebbian
        context.save_for_backward(input, weight, weight_fa, bias)

        return output

    @staticmethod
    def backward(context, grad_output):
        #the `weight` variable should always go unused as there is no backpropagation
        input, weight, weight_fa, bias= context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa.to(grad_output.device))
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        #must return a grad for each input into `forward`. This means there are useless empty ones, like grad_weight_fa
        return grad_input, grad_weight, grad_weight_fa, grad_bias
    
class WMPerceptron(MultiLayerPerceptron):

  def __init__(self, **kwargs):

    super().__init__(**kwargs)
    #Create feedback weights
    self.lin1.weight_fa = Variable(torch.FloatTensor(self.num_hidden, self.num_inputs), requires_grad=False)
    self.lin2.weight_fa = Variable(torch.FloatTensor(self.num_outputs, self.num_hidden), requires_grad=False)
    #Initiate them randomly. In standard FA these are constant and never changed
    torch.nn.init.kaiming_uniform_(self.lin1.weight_fa)
    torch.nn.init.kaiming_uniform_(self.lin2.weight_fa)


  def forward(self, X, y=None):

    h = LinearWMFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1.weight_fa,
        self.lin1.bias
    )

    y_pred = LinearWMFunction.apply(
        h,
        self.lin2.weight,
        self.lin2.weight_fa,
        self.lin2.bias
    )

    return y_pred
  
  #Mirror function, converted from https://github.com/makrout/Deep-Learning-without-Weight-Transport/blob/master/fcnn/FCNN_WM.py
  def mirror(self, X, m_lr, noise_amplitude):

    #randomly stimulate each layer
    noise_in = noise_amplitude * (torch.randn_like(X.reshape(-1, self.num_inputs)) - 0.5)
    #output of random stim
    noise_out = noise_in.mm(self.lin1.weight.t())

    #update FA weights via hebbian
    grad_fa = m_lr * noise_out.t().mm(noise_in)

    grad_fa = grad_fa / len(noise_in) # average across batch
    # center around 0
    grad_fa = grad_fa - grad_fa.mean(axis=0)
    self.lin1.weight_fa += grad_fa

    # if self.bias is not None:
    #     noise_out += self.bias.unsqueeze(0).expand_as(noise_out)

    noise_in = noise_amplitude * (torch.randn_like(noise_out) - 0.5)
    noise_out = noise_in.mm(self.lin2.weight.t())
    grad_fa = m_lr * noise_out.t().mm(noise_in)
    # average across batch
    grad_fa = grad_fa / len(noise_in) 
    # center around 0
    grad_fa = grad_fa - grad_fa.mean(axis=0)
    self.lin2.weight_fa += grad_fa
    # if self.bias is not None:
    #     noise_out += self.bias.unsqueeze(0).expand_as(noise_out)



    return
