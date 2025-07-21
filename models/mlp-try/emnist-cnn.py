import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, RandomRotation
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = Compose([
    RandomRotation(degrees=0),  
    ToTensor(),
    lambda x: x.transpose(1, 2).flip(1)  
])

train_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='byclass',
    train=True,
    download=True,
    transform=transform
) #Change this once data is downloaded in the data folder.

test_dataset = torchvision.datasets.EMNIST(
    root='./data',
    split='byclass', #currenlty contains upper case letters, lower case letters and digits
    train=False,
    download=True,
    transform=transform
)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        ) # Three convolutional layers maybe an overkill?
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
model = EMNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=running_loss/(total/batch_size), 
                         acc=100.*correct/total)
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

num_epochs = 10
best_acc = 0.0

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test_model(model, test_loader, criterion)
    
    scheduler.step(test_acc)
    
    print(f"Epoch {epoch}: "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    if test_acc > best_acc:
        best_acc = test_acc

print(f"\nBest Test Accuracy: {best_acc:.2f}%")



model.eval()
image, label = test_dataset[1055]  


image_tensor = image.unsqueeze(0).to(device)


with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()


emnist_mapping = train_dataset.classes  
true_char = emnist_mapping[label]
predicted_char = emnist_mapping[predicted_class]


plt.imshow(image.squeeze(0).cpu().numpy(), cmap='gray')
plt.title(f"True: {true_char} | Predicted: {predicted_char}")
plt.axis('off')
plt.show()