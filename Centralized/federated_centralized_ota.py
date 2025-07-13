import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("images", exist_ok=True)

# 1. Define the model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))

# 2. Dataset and clients
transform = transforms.ToTensor()
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

client_data = random_split(full_train, [30000, 30000])
client_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_data]
test_loader = DataLoader(test_data, batch_size=128)

# 3. Local Training
def local_train(model, loader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

# 4. OTA-style Aggregation (weight-level summation)
def ota_aggregate(models):
    agg_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for name, param in agg_model.named_parameters():
            stacked = torch.stack([m.state_dict()[name] for m in models])
            param.copy_(torch.sum(stacked, dim=0) / len(models))  # OTA: sum then average
    return agg_model

# 5. Evaluate
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 6. Main Simulation
NUM_ROUNDS = 5
NUM_CLIENTS = 2

global_model = MLP().to(device)
accuracy_over_time = []

for round in range(NUM_ROUNDS):
    print(f"\n📡 OTA Federated Round {round + 1}")
    local_models = []

    # Each client trains on their data
    for i in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model).to(device)
        local_train(local_model, client_loaders[i], epochs=1)
        local_models.append(local_model)

    # Simulate OTA-style weight summation
    global_model = ota_aggregate(local_models)

    # Evaluate global model
    acc = evaluate(global_model)
    accuracy_over_time.append(acc)
    print(f"✅ Accuracy after Round {round + 1}: {acc*100:.2f}%")

# 7. Plot
plt.figure(figsize=(6, 4))
plt.plot(range(1, NUM_ROUNDS + 1), [a*100 for a in accuracy_over_time], marker='o', label='OTA-FL')
plt.title("OTA Federated Learning Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.ylim(80, 100)
plt.grid(True)
plt.legend()
plt.savefig("images/ota_federated_accuracy.png")
plt.show()
