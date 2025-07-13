import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
import threading

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Simple MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))

# Load MNIST and split for 2 clients
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

# Aggregation (FedAvg)
def fedavg(models):
    new_model = deepcopy(models[0])
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            param.copy_(sum(m.state_dict()[name] for m in models) / len(models))
    return new_model

# Multithreaded training
NUM_CLIENTS = 2
global_model = MLP().to(device)
NUM_ROUNDS = 3
client_models = [None] * NUM_CLIENTS  # shared list to store trained models

for round in range(NUM_ROUNDS):
    print(f"\n🔁 Round {round+1}")

    threads = []

    # Define training for each client in a thread
    def train_client(i):
        local_model = deepcopy(global_model).to(device)
        local_train(local_model, client_loaders[i], epochs=1)
        client_models[i] = local_model

    # Start threads
    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=train_client, args=(i,))
        t.start()
        threads.append(t)

    # Wait for all to finish
    for t in threads:
        t.join()

    # Aggregate models
    global_model = fedavg(client_models)

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
