# Decentralized FL with OTA-style noisy model transmission (Ring Topology)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Parameters
NUM_CLIENTS = 5
NUM_ROUNDS = 5
EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
NOISE_STD = 0.02  # Gaussian noise std
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 32, 10)
        )

    def forward(self, x):
        return self.layer(x)

# Noise Injection (simulate OTA transmission)
def add_noise_to_model(model, noise_std):
    noisy_model = deepcopy(model)
    with torch.no_grad():
        for param in noisy_model.parameters():
            noise = torch.normal(0, noise_std, size=param.data.size()).to(param.device)
            param.add_(noise)
    return noisy_model

# Weighted averaging
def average_models(models):
    new_model = deepcopy(models[0])
    with torch.no_grad():
        for key in new_model.state_dict().keys():
            avg = torch.stack([model.state_dict()[key].float() for model in models], 0).mean(0)
            new_model.state_dict()[key].copy_(avg)
    return new_model

# Local training
def train_local(model, loader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Evaluation
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

# Prepare data
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)

# Split train data across clients
indices = list(range(len(train_data)))
random.shuffle(indices)
split_size = len(indices) // NUM_CLIENTS
client_loaders = [DataLoader(Subset(train_data, indices[i*split_size:(i+1)*split_size]), 
                             batch_size=BATCH_SIZE, shuffle=True)
                  for i in range(NUM_CLIENTS)]

# Initialize models
clients = [CNN().to(DEVICE) for _ in range(NUM_CLIENTS)]
acc_history = []

# Training rounds
for r in range(NUM_ROUNDS):
    print(f"\n--- Round {r+1} ---")
    new_clients = []

    for i in range(NUM_CLIENTS):
        local_model = deepcopy(clients[i])
        train_local(local_model, client_loaders[i], epochs=EPOCHS)

        # Receive noisy model from neighbor
        neighbor_model = deepcopy(clients[(i + 1) % NUM_CLIENTS])
        noisy_neighbor = add_noise_to_model(neighbor_model, NOISE_STD)

        # Average own model with noisy neighbor
        updated_model = average_models([local_model, noisy_neighbor])
        new_clients.append(updated_model)

    clients = new_clients
    global_model = average_models(clients)
    acc = evaluate(global_model, test_loader)
    acc_history.append(acc)
    print(f"Accuracy: {acc:.2f}%")

# Save graph
os.makedirs("images", exist_ok=True)
plt.plot(range(1, NUM_ROUNDS+1), acc_history, marker='o')
plt.xlabel("Round")
plt.ylabel("Test Accuracy (%)")
plt.title("Decentralized FL with OTA-style Noise")
plt.grid(True)
plt.savefig("images/decentralized_ota_noise_accuracy.png")
plt.show()

