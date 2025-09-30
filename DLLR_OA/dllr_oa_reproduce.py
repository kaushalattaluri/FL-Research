#!/usr/bin/env python3
"""
dllr_oa_reproduce.py

Minimal reproduction of DLLR-OA style decentralized OTA federated learning.
Simulates N clients communicating over a graph using OTA-style analog aggregation.

Produces:
 - results.csv with per-round accuracy & simple stats
 - accuracy plot saved to images/accuracy_plot.png

Notes:
 - This is a simulator (no real RF). OTA is simulated as vector superposition + Gaussian channel noise.
 - The code is intentionally explicit and easy to extend (clusters, DP adaptation, error-feedback).
"""

import os
import time
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# ---------------------------
# Config / hyperparameters
# ---------------------------
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# Simulation & dataset
NUM_CLIENTS = 10
ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9

# OTA / DLLR-OA specifics
TOP_K_RATIO = 0.05   # fraction of parameters to keep in top-k sparsification
DP_SIGMA = 1e-3      # Gaussian DP noise std on transmitted vector
CHANNEL_NOISE_STD = 1e-3  # channel (air) noise std
POWER_SCALE = True   # whether to normalize transmitted vector energy (simple scaling)
NEIGHBOR_GRAPH = "erdos_renyi"  # 'ring' | 'mesh' | 'erdos_renyi'
ERDOS_P = 0.2

# Logging / output
OUTPUT_DIR = "dllr_oa_outputs"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
PLOT_PATH = os.path.join(IMAGES_DIR, "accuracy_plot.png")

os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------
# Simple CNN model for CIFAR-10 (small)
# ---------------------------

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3,32,32)):
        super(SmallCNN, self).__init__()
        # define conv/pool layers on self (so parameters are registered)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial dims once

        # compute flattened size by a dummy forward (batch=1)
        with torch.no_grad():
            c, h, w = input_shape
            dummy = torch.zeros(1, c, h, w)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            # print debug info for computed dummy tensor
            print(">>> [_init_] after conv/pool dummy shape:", tuple(x.shape))
            flat_size = x.numel()  # total elements for batch=1 (1*C*H*W)
            print(">>> [_init_] computed flat_size:", flat_size)

        # build FC layers using computed flat_size
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, debug=False):
        if debug:
            print(">>> [forward] input shape:", tuple(x.shape))
        x = F.relu(self.conv1(x))
        if debug:
            print(">>> [forward] after conv1:", tuple(x.shape))
        x = F.relu(self.conv2(x))
        if debug:
            print(">>> [forward] after conv2:", tuple(x.shape))
        x = self.pool(x)
        if debug:
            print(">>> [forward] after pool:", tuple(x.shape))
        # flattened vector length per sample (C*H*W)
        flattened_per_sample = x.shape[1] * x.shape[2] * x.shape[3]
        if debug:
            print(">>> [forward] flattened_per_sample:", flattened_per_sample)
            print(">>> [forward] fc1.in_features:", self.fc1.in_features)
        # sanity check
        if flattened_per_sample != self.fc1.in_features:
            raise RuntimeError(
                f"Shape mismatch: flattened_per_sample={flattened_per_sample} != fc1.in_features={self.fc1.in_features}"
            )
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Utilities: model <-> vector conversion
# ---------------------------
def model_to_vector(model):
    """Flatten model parameters into a single 1D numpy array (float32)."""
    with torch.no_grad():
        vecs = []
        for p in model.parameters():
            vecs.append(p.data.cpu().numpy().ravel())
        return np.concatenate(vecs).astype(np.float32)

def vector_to_model(model, vec):
    """Load a flattened numpy vector into model parameters (in-place)."""
    with torch.no_grad():
        pointer = 0
        for p in model.parameters():
            numel = p.numel()
            slice_ = vec[pointer:pointer+numel].reshape(p.size())
            p.data.copy_(torch.from_numpy(slice_).to(p.data.device))
            pointer += numel

def model_param_size(model):
    """Return total number of parameters (int)."""
    return sum(p.numel() for p in model.parameters())

# ---------------------------
# Data partitioning (Dirichlet non-IID)
# ---------------------------
def create_dirichlet_partitions(dataset, num_clients, alpha=0.5, min_size=10):
    """
    Partition dataset indices into num_clients non-iid parts using Dirichlet distribution.
    Returns list of index lists per client.
    """
    # dataset.targets assumes torchvision dataset with .targets (list)
    labels = np.array(dataset.targets)
    num_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            n = len(idx_by_class[c])
            if n == 0:
                continue
            proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
            # scale proportions to counts
            counts = (proportions * n).astype(int)
            # fix rounding issues by assigning remaining
            while counts.sum() < n:
                counts[np.argmax(proportions)] += 1
            # sample and split
            idxs = np.random.permutation(idx_by_class[c])
            pointer = 0
            for i in range(num_clients):
                cnt = counts[i]
                if cnt > 0:
                    chosen = idxs[pointer:pointer+cnt].tolist()
                    client_indices[i].extend(chosen)
                    pointer += cnt
        # ensure min size
        ok = all(len(idx) >= min_size for idx in client_indices)
        if ok:
            return client_indices

# ---------------------------
# Build neighbor graph
# ---------------------------
def build_graph(num_clients, mode="erdos_renyi", p=0.2):
    neighbors = {i: set() for i in range(num_clients)}
    if mode == "ring":
        for i in range(num_clients):
            neighbors[i].add((i-1) % num_clients)
            neighbors[i].add((i+1) % num_clients)
    elif mode == "mesh":
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    neighbors[i].add(j)
    elif mode == "erdos_renyi":
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                if np.random.rand() < p:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        # ensure connectivity per node by at least one neighbor
        for i in range(num_clients):
            if len(neighbors[i]) == 0:
                # connect to a random other node
                j = np.random.choice([x for x in range(num_clients) if x != i])
                neighbors[i].add(j)
                neighbors[j].add(i)
    else:
        raise ValueError("Unknown graph mode")
    # convert sets to sorted lists
    return {i: sorted(list(neighbors[i])) for i in range(num_clients)}

# ---------------------------
# Sparsification (top-k)
# ---------------------------
def top_k_sparsify(vec, k):
    """Keep top k largest magnitude entries, zero others. vec: numpy array."""
    if k <= 0 or k >= vec.size:
        return vec.copy()
    absv = np.abs(vec)
    if isinstance(k, float) and 0 < k < 1:
        k = int(k * vec.size)
    idx = np.argpartition(-absv, k-1)[:k]
    mask = np.zeros_like(vec, dtype=bool)
    mask[idx] = True
    out = np.zeros_like(vec)
    out[mask] = vec[mask]
    return out

# ---------------------------
# Local training utility
# ---------------------------
def local_train(model, train_loader, epochs, device, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_fn(out, target)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

# ---------------------------
# Main simulation
# ---------------------------
def main():
    # -----------------------
    # Prepare dataset
    # -----------------------
    print("Preparing CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # create non-iid partitions
    partitions = create_dirichlet_partitions(cifar_train, NUM_CLIENTS, alpha=0.5)
    client_loaders = []
    for i in range(NUM_CLIENTS):
        idx = partitions[i]
        subset = torch.utils.data.Subset(cifar_train, idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        client_loaders.append(loader)

    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=256, shuffle=False, num_workers=0)

    # -----------------------
    # Initialize client models
    # -----------------------
    print("Initializing client models...")
    clients = []
    for i in range(NUM_CLIENTS):
        m = SmallCNN().to(DEVICE)
        clients.append(m)

    # initialize global param vector (common init)
    global_model = SmallCNN().to(DEVICE)
    print("global_model type:", type(global_model))
    try:
        params = list(global_model.parameters())
        print("number of parameter tensors:", len(params))
        for i, p in enumerate(params[:10]):
            print(f" param[{i}] shape: {tuple(p.shape)} | requires_grad: {p.requires_grad}")
    except Exception as e:
        print("Error while listing parameters:", e)

# Now safe check before calling model_to_vector
    if len(list(global_model.parameters())) == 0:
        raise RuntimeError("Model has 0 parameters — check SmallCNN definition and instantiation.")
    base_vec = model_to_vector(global_model)
    for m in clients:
        vector_to_model(m, base_vec.copy())

    param_size = model_param_size(global_model)
    print(f"Model parameter size: {param_size}")

    # neighbor graph
    neighbors = build_graph(NUM_CLIENTS, mode=NEIGHBOR_GRAPH, p=ERDOS_P)
    for i in range(NUM_CLIENTS):
        print(f"Client {i} neighbors: {neighbors[i]}")

    # logging storage
    results = []
    global_accs = []

    # For each round:
    for r in trange(ROUNDS, desc="Rounds"):
        round_start = time.time()
        # store transmitted vectors for each client this round
        transmitted = [None] * NUM_CLIENTS

        # Each client does local training starting from its current model copy
        for i in range(NUM_CLIENTS):
            local_train(clients[i], client_loaders[i], LOCAL_EPOCHS, DEVICE, lr=LR)

        # compute deltas (local update w.r.t previous param)
        for i in range(NUM_CLIENTS):
            # delta = new model - previous global model vector (we consider previous local baseline as base_vec)
            # to mimic DLLR-OA: we'll use a "local delta" from previous round global baseline
            local_vec = model_to_vector(clients[i])
            delta = local_vec - base_vec  # numpy arrays
            # sparsify (top-k)
            k = int(TOP_K_RATIO * delta.size)
            if k < 1:
                k = 1
            sparse_delta = top_k_sparsify(delta, k)
            # power scaling: normalize energy (simple)
            if POWER_SCALE:
                norm = np.linalg.norm(sparse_delta)
                if norm > 0:
                    sparse_delta = sparse_delta / norm  # unit-energy
            # add gaussian DP noise
            noisy = sparse_delta + np.random.normal(scale=DP_SIGMA, size=sparse_delta.shape).astype(np.float32)
            transmitted[i] = noisy

        # simulate OTA aggregation for each client via neighbors
        aggregated = [None] * NUM_CLIENTS
        for i in range(NUM_CLIENTS):
            neigh = neighbors[i] + [i]  # include self usually
            # analog sum/average with channel noise per-dimension
            sum_vec = np.zeros(param_size, dtype=np.float32)
            for j in neigh:
                sum_vec += transmitted[j]
            # average
            agg = sum_vec / len(neigh)
            # channel noise
            agg += np.random.normal(scale=CHANNEL_NOISE_STD, size=agg.shape).astype(np.float32)
            # (optional) recovery scaling: if we normalized transmit vectors, undo by multiplying by typical norm
            # Here we assume unbiased averaging is enough
            aggregated[i] = agg

        # Apply aggregated updates to local models (mix with local)
        # Here we do a simple mixing: new_theta = old_theta + gamma * aggregated
        gamma = 1.0  # mixing coefficient; tune as hyperparam
        for i in range(NUM_CLIENTS):
            cur_vec = model_to_vector(clients[i])
            new_vec = cur_vec + gamma * aggregated[i]
            vector_to_model(clients[i], new_vec)

        # update base_vec (baseline) to be average of all client models (synchronous view)
        avg_vec = np.mean([model_to_vector(clients[i]) for i in range(NUM_CLIENTS)], axis=0)
        base_vec = avg_vec.copy()  # next round baseline

        # evaluate: compute average test acc across clients (evaluate each client model on full test set)
        accs = []
        losses = []
        for i in range(NUM_CLIENTS):
            acc, loss = evaluate_model(clients[i], test_loader, DEVICE)
            accs.append(acc)
            losses.append(loss)
        mean_acc = float(np.mean(accs))
        mean_loss = float(np.mean(losses))
        round_time = time.time() - round_start

        # compute simple comm cost proxy: k elements per client per round
        comm_cost = NUM_CLIENTS * int(TOP_K_RATIO * param_size)
        # energy_used proxy: comm_cost * (some factor); here we record comm_cost as energy proxy
        energy_used = comm_cost

        results.append({
            "run_id": 0,
            "num_clients": NUM_CLIENTS,
            "local_epochs": LOCAL_EPOCHS,
            "top_k_ratio": TOP_K_RATIO,
            "dp_sigma": DP_SIGMA,
            "snr": None,
            "round": r+1,
            "accuracy": mean_acc,
            "loss": mean_loss,
            "comm_cost": comm_cost,
            "energy_used": energy_used,
            "round_time": round_time
        })
        global_accs.append(mean_acc)

        if (r+1) % PRINT_EVERY == 0:
            print(f"Round {r+1} | mean acc: {mean_acc:.2f} | loss: {mean_loss:.4f} | comm_cost: {comm_cost}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved results to {CSV_PATH}")

    # Save accuracy plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, ROUNDS+1), global_accs, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Mean Test Accuracy (%) across clients")
    plt.title("DLLR-OA Reproduction: Mean Accuracy vs Rounds")
    plt.grid(True)
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"Saved plot to {PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    main()
