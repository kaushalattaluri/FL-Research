import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("images", exist_ok=True)

# ---- 1. Model ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))

# ---- 2. Data Split by Task ----
def get_task_dataset(task_digits):
    def filter_digits(dataset, digits):
        indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
        return Subset(dataset, indices)

    transform = transforms.ToTensor()
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    return (filter_digits(train, task_digits), filter_digits(test, task_digits))

# ---- 3. Training and Evaluation ----
def train(model, dataloader, epochs=3, lr=0.01):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ---- 4. EWC ----
class EWC:
    def __init__(self, model, dataset, lambda_=1000):
        self.model = model
        self.lambda_ = lambda_
        self.params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.fisher = self._estimate_fisher(dataset)

    def _estimate_fisher(self, dataset):
        fisher = {}
        model = self.model
        model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for n, p in model.named_parameters():
            fisher[n] = torch.zeros_like(p)
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            for n, p in model.named_parameters():
                fisher[n] += p.grad.data.clone().pow(2)

        for n in fisher:
            fisher[n] /= len(loader)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ * loss

def train_with_ewc(model, dataset, ewc_obj, epochs=3, lr=0.01):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            ce_loss = nn.CrossEntropyLoss()(out, y)
            ewc_loss = ewc_obj.penalty(model)
            loss = ce_loss + ewc_loss
            loss.backward()
            opt.step()

# ---- 5. Run Experiment ----
# Get tasks
task1_digits = [0, 1, 2, 3, 4]
task2_digits = [5, 6, 7, 8, 9]

task1_train, task1_test = get_task_dataset(task1_digits)
task2_train, task2_test = get_task_dataset(task2_digits)

# DataLoaders
task1_loader = DataLoader(task1_train, batch_size=64, shuffle=True)
task2_loader = DataLoader(task2_train, batch_size=64, shuffle=True)
task1_test_loader = DataLoader(task1_test, batch_size=128)
task2_test_loader = DataLoader(task2_test, batch_size=128)

# ---- 6. Train Without EWC (Catastrophic Forgetting) ----
print("➡️ Training without EWC")
model_plain = MLP().to(device)
train(model_plain, task1_loader, epochs=3)
acc_task1_before = evaluate(model_plain, task1_test_loader)
print(f"Accuracy on Task 1 before Task 2: {acc_task1_before*100:.2f}%")

train(model_plain, task2_loader, epochs=3)
acc_task1_after = evaluate(model_plain, task1_test_loader)
acc_task2 = evaluate(model_plain, task2_test_loader)
print(f"Accuracy on Task 1 after Task 2: {acc_task1_after*100:.2f}%")
print(f"Accuracy on Task 2: {acc_task2*100:.2f}%")

# ---- 7. Train With EWC ----
print("\n➡️ Training with EWC")
model_ewc = MLP().to(device)
train(model_ewc, task1_loader, epochs=3)
ewc = EWC(model_ewc, task1_train)
acc_ewc_task1_before = evaluate(model_ewc, task1_test_loader)

train_with_ewc(model_ewc, task2_train, ewc, epochs=3)
acc_ewc_task1_after = evaluate(model_ewc, task1_test_loader)
acc_ewc_task2 = evaluate(model_ewc, task2_test_loader)

print(f"Accuracy on Task 1 before Task 2 (EWC): {acc_ewc_task1_before*100:.2f}%")
print(f"Accuracy on Task 1 after Task 2 (EWC): {acc_ewc_task1_after*100:.2f}%")
print(f"Accuracy on Task 2 (EWC): {acc_ewc_task2*100:.2f}%")

# ---- 8. Plot Results ----
labels = ['Task 1 Before', 'Task 1 After', 'Task 2']
no_ewc = [acc_task1_before, acc_task1_after, acc_task2]
with_ewc = [acc_ewc_task1_before, acc_ewc_task1_after, acc_ewc_task2]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, no_ewc, width, label='No EWC')
plt.bar(x + width/2, with_ewc, width, label='With EWC')
plt.ylabel('Accuracy')
plt.title('Continual Learning: Catastrophic Forgetting vs EWC')
plt.xticks(x, labels)
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(True)
plt.savefig("images/ewc_vs_noewc_accuracy.png")
plt.show()
