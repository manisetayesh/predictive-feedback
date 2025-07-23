import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets_loader import load_extended_mnist
from cnns import EMNIST_CNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_dataset, test_dataset = load_extended_mnist()


batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    
model = EMNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

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