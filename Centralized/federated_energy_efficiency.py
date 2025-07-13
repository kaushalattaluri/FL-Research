import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))

# Load MNIST & split among 5 clients
transform = transforms.ToTensor()
full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

NUM_CLIENTS = 5
client_sizes = [12000] * 5
client_datasets = random_split(full_train, client_sizes)
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

# Aggregation
def fedavg(models):
    new_model = deepcopy(models[0])
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            param.copy_(sum(m.state_dict()[name] for m in models) / len(models))
    return new_model

# Energy-based client selection
def get_client_energy_cost():
    # Simulate energy needed for each client to train + send weights
    return [random.uniform(0.1, 1.5) for _ in range(NUM_CLIENTS)]

# Federated training
NUM_ROUNDS = 5
global_model = MLP().to(device)

ENERGY_BUDGET_PER_ROUND = 3.0  # max energy allowed per round

for round in range(NUM_ROUNDS):
    print(f"\n⚡ Round {round+1}")
    local_models = []
    energy_costs = get_client_energy_cost()

    selected_clients = []
    total_energy = 0.0

    print("Client Energy Costs:", [f"{e:.2f}" for e in energy_costs])

    # Select clients greedily under budget
    for i in sorted(range(NUM_CLIENTS), key=lambda x: energy_costs[x]):
        if total_energy + energy_costs[i] <= ENERGY_BUDGET_PER_ROUND:
            selected_clients.append(i)
            total_energy += energy_costs[i]

    print("✅ Selected Clients:", selected_clients)
    print(f"🔋 Total Energy Used: {total_energy:.2f} units")

    # Local training for selected clients
    for i in selected_clients:
        local_model = deepcopy(global_model).to(device)
        local_train(local_model, client_loaders[i], epochs=1)
        local_models.append(local_model)

    # Only aggregate if we have at least one participant
    if local_models:
        global_model = fedavg(local_models)

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
    print(f"🎯 Accuracy: {acc*100:.2f}%")
