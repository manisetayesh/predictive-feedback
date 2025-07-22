import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import random
from collections import defaultdict

learning_rate = 0.003
meta_step_size = 0.25

inner_batch_size = 25
eval_batch_size = 25

meta_iters = 4000
eval_iters = 5
inner_iters = 4

eval_interval = 1
train_shots = 20
shots = 5
classes = 5





transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    lambda x: 1 - x  
])

omniglot_train = datasets.Omniglot(root="./data", background=True, download=True, transform=transform)


class_to_indices = defaultdict(list)
for idx, (img, label) in enumerate(omniglot_train):
    class_to_indices[label].append(idx)

print(class_to_indices)
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layer1 = ConvBNReLU(1, 64)
        self.layer2 = ConvBNReLU(64, 64)
        self.layer3 = ConvBNReLU(64, 64)
        self.layer4 = ConvBNReLU(64, 64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 2 * 2, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def reptile(model, nb_iterations: int, sample_task: Callable, perform_k_training_steps: Callable, k=1, epsilon=0.1):
    for _ in tqdm(range(nb_iterations)):

        task = sample_task()
        phi_tilde = perform_k_training_steps(copy.deepcopy(model), task, k)

       
        with torch.no_grad():
            for p, g in zip(model.parameters(), phi_tilde):
                p += epsilon * (g - p)


@torch.no_grad()
def sample_task(n_way=5, k_shot=5, q_queries=15):
    selected_classes = random.sample(list(class_to_indices.keys()), n_way)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for i, cls in enumerate(selected_classes):
        indices = random.sample(class_to_indices[cls], k_shot + q_queries)
        support_indices = indices[:k_shot]
        query_indices = indices[k_shot:]

        support_x.extend([omniglot_train[idx][0] for idx in support_indices])
        support_y.extend([i] * k_shot)

        query_x.extend([omniglot_train[idx][0] for idx in query_indices])
        query_y.extend([i] * q_queries)

    support_x = torch.stack(support_x)
    support_y = torch.tensor(support_y)

    query_x = torch.stack(query_x)
    query_y = torch.tensor(query_y)

    loss_fct = nn.CrossEntropyLoss()
    return (support_x, support_y, query_x, query_y, loss_fct)


def perform_k_training_steps(model, task, k, batch_size=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    support_x, support_y, query_x, query_y, loss_fct = task
    data_size = support_x.size(0)

    for _ in range(k * data_size // batch_size):
        ind = torch.randperm(data_size)[:batch_size]
        x_batch = support_x[ind]
        y_batch = support_y[ind]

        logits = model(x_batch)
        loss = loss_fct(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return [p.clone() for p in model.parameters()]



if __name__ == "__main__":
    model = Model(classes)
    reptile(model, meta_iters, sample_task, perform_k_training_steps)

    
    model_copy = copy.deepcopy(model)
    task = sample_task()
    support_x, support_y, query_x, query_y, _ = task

    perform_k_training_steps(model_copy, task, 32)

    with torch.no_grad():
        preds = model_copy(query_x).argmax(dim=1)
        accuracy = (preds == query_y).float().mean().item()
        print(f"Accuracy on new task: {accuracy * 100:.2f}%")


'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset:
    def __init__(self, training):
        split = "train" if training else "test"
        background = (split == "train")

        ds = datasets.Omniglot(
            root='./data',
            background=background,
            download=True,
        )

        self.data = {}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
        ])

        for img, label in ds:
            img = transform(img)  # (1,28,28) tensor float32
            label = str(label)
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(img)

        self.labels = list(self.data.keys())

    def get_mini_dataset(self, batch_size, repetitions, shots, num_classes, split=False):
        total_shots = num_classes * shots
        temp_labels = np.zeros(shape=(total_shots))
        temp_images = np.zeros(shape=(total_shots, 28, 28, 1), dtype=np.float32)

        if split:
            test_labels = np.zeros(shape=(num_classes))
            test_images = np.zeros(shape=(num_classes, 28, 28, 1), dtype=np.float32)

        label_subset = random.choices(self.labels, k=num_classes)

        for class_idx, class_obj in enumerate(label_subset):
            temp_labels[class_idx * shots : (class_idx + 1) * shots] = class_idx
            images_list = self.data[class_obj]
            images_numpy = [img.permute(1, 2, 0).numpy() if torch.is_tensor(img) else img for img in images_list]

            if split:
                test_labels[class_idx] = class_idx
                images_to_split = random.choices(images_numpy, k=shots + 1)
                test_images[class_idx] = images_to_split[-1]
                temp_images[class_idx * shots : (class_idx + 1) * shots] = images_to_split[:-1]
            else:
                temp_images[class_idx * shots : (class_idx + 1) * shots] = random.choices(images_numpy, k=shots)

        images_tensor = torch.tensor(temp_images).permute(0, 3, 1, 2)  # (N,1,28,28)
        labels_tensor = torch.tensor(temp_labels, dtype=torch.long)

        dataset = TensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        def repeated_loader():
            for _ in range(repetitions):
                for batch in dataloader:
                    yield batch

        if split:
            return repeated_loader(), test_images, test_labels
        return repeated_loader()

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpleModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layer1 = ConvBNReLU(1, 64)
        self.layer2 = ConvBNReLU(64, 64)
        self.layer3 = ConvBNReLU(64, 64)
        self.layer4 = ConvBNReLU(64, 64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 2 * 2, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Instantiate datasets and model
train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

model = SimpleModel(classes=classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

training = []
testing = []

for meta_iter in range(meta_iters):
    frac_done = meta_iter / meta_iters
    cur_meta_step_size = (1 - frac_done) * meta_step_size

    old_vars = copy.deepcopy(model.state_dict())

    mini_dataset = train_dataset.get_mini_dataset(inner_batch_size, inner_iters, train_shots, classes)

    model.train()
    for images, labels in mini_dataset:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

    new_vars = model.state_dict()

    with torch.no_grad():
        for key in old_vars:
            old_vars[key].data.copy_(
                old_vars[key].data + (new_vars[key].data - old_vars[key].data) * cur_meta_step_size
            )

    model.load_state_dict(old_vars)

    if meta_iter % eval_interval == 0:
        accuracies = []

        for dataset in (train_dataset, test_dataset):
            train_set, test_images_np, test_labels_np = dataset.get_mini_dataset(
                eval_batch_size, eval_iters, shots, classes, split=True
            )

            test_images = torch.tensor(test_images_np).permute(0, 3, 1, 2).to(device)
            test_labels = torch.tensor(test_labels_np, dtype=torch.long).to(device)

            old_vars_eval = copy.deepcopy(model.state_dict())

            model.train()
            for images, labels in train_set:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_preds = model(test_images)
                predicted_labels = torch.argmax(test_preds, dim=1)

                num_correct = (predicted_labels == test_labels).sum().item()
                accuracy = num_correct / classes
                accuracies.append(accuracy)

            model.load_state_dict(old_vars_eval)

        training.append(accuracies[0])
        testing.append(accuracies[1])
        if meta_iter % 100 == 0:
            print(f"batch {meta_iter}: train={accuracies[0]:.6f} test={accuracies[1]:.6f}")
            '''