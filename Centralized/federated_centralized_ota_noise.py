import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))

# Load MNIST & split for 2 clients
transform = transforms.ToTensor()
full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

client_datasets = random_split(full_train, [30000, 30000])
client_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_datasets]
test_loader = DataLoader(test_data, batch_size=128)

# Local training
def local_train(model, dataloader, epochs=1):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

# OTA-like noisy aggregation
def noisy_aggregate(models, noise_std=0.01):
    new_model = deepcopy(models[0])
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            # Stack parameters from all models
            stacked = torch.stack([m.state_dict()[name] for m in models])
            summed = torch.sum(stacked, dim=0)
            
            # Add Gaussian noise
            noise = torch.randn_like(summed) * noise_std
            averaged = (summed + noise) / len(models)

            param.copy_(averaged)
    return new_model

# Federated Rounds
NUM_CLIENTS = 2
NUM_ROUNDS = 5
global_model = MLP().to(device)

for round in range(NUM_ROUNDS):
    print(f"\n📡 OTA-FL Round {round+1}")
    local_models = []

    for i in range(NUM_CLIENTS):
        local_model = deepcopy(global_model).to(device)
        local_train(local_model, client_loaders[i], epochs=1)
        local_models.append(local_model)

    # OTA Aggregation with noise
    global_model = noisy_aggregate(local_models, noise_std=0.02)  # Try changing noise_std

    # Evaluate
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = global_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"✅ Accuracy: {acc*100:.2f}%")
