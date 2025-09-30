import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmallCNNDebug(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3,32,32)):
        super(SmallCNNDebug, self).__init__()
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

def model_to_vector(model):
    with torch.no_grad():
        vecs = []
        for p in model.parameters():
            arr = p.data.cpu().numpy().ravel()
            vecs.append(arr)
        if len(vecs) == 0:
            raise RuntimeError("model_to_vector: model has 0 parameter arrays")
        return np.concatenate(vecs).astype(np.float32)

def main():
    print("Device:", DEVICE)
    model = SmallCNNDebug().to(DEVICE)
    print("Model instantiated:", type(model))
    params = list(model.parameters())
    print("Number of parameter tensors:", len(params))
    total_params = sum(p.numel() for p in params)
    print("Total number of parameters:", total_params)
    for i, p in enumerate(params[:8]):
        print(f" param[{i}] shape: {tuple(p.shape)} requires_grad={p.requires_grad}")

    # Forward pass with debug prints
    dummy_input = torch.randn(4, 3, 32, 32).to(DEVICE)  # batch size 4
    try:
        out = model(dummy_input, debug=True)
        print("Forward output shape:", tuple(out.shape))  # expected (4, 10)
    except RuntimeError as e:
        print("RuntimeError during forward:", e)
        return

    # Flatten to vector and show its length
    vec = model_to_vector(model)
    print("Flattened parameter vector length:", vec.shape[0])

    print("Sanity check passed.")

if __name__ == "__main__":
    main()