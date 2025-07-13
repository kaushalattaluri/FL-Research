import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc2(self.relu(self.fc1(x)))

# Load and Split MNIST
transform = transforms.ToTensor()
full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

client_data = torch.utils.data.random_split(full_train, [30000, 30000])
client_loaders = [torch.utils.data.DataLoader(d, batch_size=64, shuffle=True) for d in client_data]
test_loader = torch.utils.data.DataLoader(testset, batch_size=128)

# Local Training
def train(model, dataloader, epochs=1):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            optimizer.step()

# Evaluation
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# FedAvg
def fed_avg(models):
    global_model = deepcopy(models[0])
    with torch.no_grad():
        for param in global_model.state_dict():
            avg = torch.stack([m.state_dict()[param].float() for m in models], dim=0).mean(dim=0)
            global_model.state_dict()[param].copy_(avg)
    return global_model

# Federated Learning Loop
NUM_ROUNDS = 5
NUM_CLIENTS = 2
EPOCHS = 1

global_model = MLP().to(device)
accuracy_history = []

for r in range(NUM_ROUNDS):
    print(f"\n--- Federated Round {r+1} ---")
    local_models = []

    for i in range(NUM_CLIENTS):
        local_model = deepcopy(global_model).to(device)
        train(local_model, client_loaders[i], epochs=EPOCHS)
        local_models.append(local_model)

    global_model = fed_avg(local_models).to(device)
    acc = evaluate(global_model)
    accuracy_history.append(acc)
    print(f"Global model accuracy: {acc:.4f}")

# Plot Accuracy
plt.plot(range(1, NUM_ROUNDS+1), accuracy_history, marker='o')
plt.title("Centralized Federated Learning Accuracy Over Rounds")
plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.savefig("images/centralized_fl_plot.png")

plt.show()
