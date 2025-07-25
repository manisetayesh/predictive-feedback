from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt 
from datasets_loader import load_mnist
from cnns import MNIST_CNN



train_data, test_data = load_mnist()

loaders = {
    'train' : DataLoader(train_data, batch_size = 100, shuffle = True),
    'test' : DataLoader(test_data, batch_size = 100, shuffle = True)
}


def backprop_learning(model, optimizer, loss_fn, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

'''Simple  hebbian learning just for testing if i can use other learning rules in the neural network '''
def hebbian_learning(model,data, target, lr=0.01):
    with torch.no_grad():
        x = F.relu(F.max_pool2d(model.conv1(data), 2))
        x = F.relu(F.max_pool2d(model.conv2_drop(model.conv2(x)), 2))
        x = x.view(-1, 320)

        y = F.relu(model.fc1(x))  

        
        dw = torch.bmm(y.unsqueeze(2), x.unsqueeze(1)) 
        dw = dw.mean(0)  
        model.fc1.weight.data += lr * dw

def evaluate_model(model, loaders, device, loss_fn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(loaders['test'].dataset)
    print(f"Test Accuracy: {acc:.4f}")
    model.train()

def train_model(model, loaders, device, optimizer, loss_fn, learning_rule, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            loss = learning_rule(model, optimizer, loss_fn, data, target)
            if loss is not None:
                total_loss += loss
        
        evaluate_model(model, loaders, device, loss_fn)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MNIST_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_fn = nn.CrossEntropyLoss()



train_model(model, loaders, device, optimizer, loss_fn, backprop_learning, epochs=10)
#hebbian
#train_model(model, loaders, device, None, None, lambda m, o, l, d, t: hebbian_learning(m, d, t), epochs=10)
data, target = test_data[15]
data = data.unsqueeze(0).to(device)
output = model(data)
pred = output.argmax(dim = 1, keepdim = True).item()
print("prediction: ", pred)

plt.imshow(data.squeeze(0).squeeze(0).cpu().numpy(), cmap = 'gray')
plt.show()