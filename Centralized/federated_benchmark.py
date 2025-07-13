import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
import time
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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

# FedAvg with noise injection
def fedavg_with_noise(models, noise_std=0.0):
    new_model = deepcopy(models[0])
    with torch.no_grad():
        for name, param in new_model.named_parameters():
            params = torch.stack([m.state_dict()[name] for m in models])
            summed = params.sum(dim=0)
            noise = torch.randn_like(summed) * noise_std
            param.copy_((summed + noise) / len(models))
    return new_model

# Local training
def local_train(model, dataloader, epochs):
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

# Get random energy cost per client
def get_client_energy_cost(num_clients):
    return [random.uniform(0.1, 1.5) for _ in range(num_clients)]

# Benchmark runner
def run_fl_benchmark(num_clients=5, local_epochs=1, num_rounds=5, noise_std=0.0, energy_aware=False):
    # Load and split MNIST
    transform = transforms.ToTensor()
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    client_sizes = [len(full_train) // num_clients] * num_clients
    client_datasets = random_split(full_train, client_sizes)
    client_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_data, batch_size=128)

    global_model = MLP().to(device)

    ENERGY_BUDGET = 3.0
    accuracy_list, energy_used_list, round_time_list = [], [], []

    for r in range(num_rounds):
        print(f"\n🔁 Round {r+1}/{num_rounds}")
        round_start = time.time()

        selected_clients = list(range(num_clients))
        energy_used = 0

        if energy_aware:
            energy_costs = get_client_energy_cost(num_clients)
            selected_clients = []
            total_energy = 0
            for i in sorted(range(num_clients), key=lambda x: energy_costs[x]):
                if total_energy + energy_costs[i] <= ENERGY_BUDGET:
                    selected_clients.append(i)
                    total_energy += energy_costs[i]
            energy_used = total_energy
            print("⚡ Energy-aware client selection:", selected_clients)
        else:
            energy_used = num_clients * 1.0  # assume 1 unit per client

        local_models = []
        for i in selected_clients:
            local_model = deepcopy(global_model).to(device)
            local_train(local_model, client_loaders[i], local_epochs)
            local_models.append(local_model)

        global_model = fedavg_with_noise(local_models, noise_std)

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
        round_end = time.time()

        accuracy_list.append(acc * 100)
        energy_used_list.append(energy_used)
        round_time_list.append(round_end - round_start)

        print(f"🎯 Accuracy: {acc*100:.2f}% | ⏱️ Time: {round_end - round_start:.2f}s | 🔋 Energy: {energy_used:.2f}")

    return accuracy_list, energy_used_list, round_time_list


if __name__ == "__main__":
    acc, energy, timing = run_fl_benchmark(
        num_clients=5,
        local_epochs=1,
        num_rounds=5,
        noise_std=0.02,
        energy_aware=True
    )

    # Plot accuracy
    import matplotlib.pyplot as plt
    rounds = list(range(1, len(acc) + 1))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(rounds, acc, marker='o')
    plt.title("Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")

    plt.subplot(1, 3, 2)
    plt.plot(rounds, energy, marker='s')
    plt.title("Energy Used per Round")
    plt.xlabel("Round")
    plt.ylabel("Energy")

    plt.subplot(1, 3, 3)
    plt.plot(rounds, timing, marker='^')
    plt.title("Time Taken per Round")
    plt.xlabel("Round")
    plt.ylabel("Seconds")

    plt.tight_layout()
    plt.savefig("images/benchmark_run.png")
    plt.show()
