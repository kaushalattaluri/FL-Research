# Decentralized Federated Learning (Ring Topology) using PyTorch
# Each client exchanges weights with neighbors and performs local averaging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import numpy as np
import os

# Configuration
NUM_CLIENTS = 5
NUM_ROUNDS = 5
EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple CNN for MNIST
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(26 * 26 * 32, 10)
        )

    def forward(self, x):
        return self.layer(x)

# Average model weights from list of models
def average_weights(models):
    avg_model = deepcopy(models[0])
    for key in avg_model.state_dict().keys():
        avg_weight = torch.stack([model.state_dict()[key].float() for model in models], 0).mean(0)
        avg_model.state_dict()[key].copy_(avg_weight)
    return avg_model

# Load MNIST data and split among clients
transform = transforms.Compose([transforms.ToTensor()])
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
indices = list(range(len(full_train)))
random.shuffle(indices)
split_size = len(indices) // NUM_CLIENTS
client_loaders = []
for i in range(NUM_CLIENTS):
    subset = Subset(full_train, indices[i*split_size:(i+1)*split_size])
    client_loaders.append(DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True))

test_loader = DataLoader(datasets.MNIST(root='./data', train=False, transform=transform), batch_size=1000)

# Training function for each client
def train_local(model, loader, epochs=1):
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

# Evaluate global model
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

# Initialize clients' models
clients = [CNN().to(DEVICE) for _ in range(NUM_CLIENTS)]
accuracy_history = []

# Main loop: simulate ring communication
for rnd in range(NUM_ROUNDS):
    print(f"\nRound {rnd + 1}")
    new_clients = []

    for i in range(NUM_CLIENTS):
        # Deep copy own model and receive neighbor's model
        local_model = deepcopy(clients[i])
        neighbor_model = deepcopy(clients[(i + 1) % NUM_CLIENTS])

        # Train local model
        train_local(local_model, client_loaders[i], epochs=EPOCHS)

        # Average with neighbor
        avg_model = average_weights([local_model, neighbor_model])
        new_clients.append(avg_model)

    # Update client models
    clients = new_clients

    # Evaluate average model for reporting
    global_model = average_weights(clients)
    acc = evaluate(global_model, test_loader)
    accuracy_history.append(acc)
    print(f"Avg Accuracy: {acc:.2f}%")

# Plot accuracy trend
os.makedirs("images", exist_ok=True)
plt.plot(range(1, NUM_ROUNDS+1), accuracy_history, marker='o')
plt.xlabel("Round")
plt.ylabel("Test Accuracy (%)")
plt.title("Decentralized FL Accuracy over Rounds")
plt.grid(True)
plt.savefig("images/decentralized_fl_accuracy.png")
plt.show()

