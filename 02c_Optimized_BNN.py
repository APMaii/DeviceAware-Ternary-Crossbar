'''
In The Name of God

Ali Pilehvar Meibody

Last Update : 28 oct 2026


02c_Optimized_BNN.py


Same MNIST network as 02b_BNN_optimization.py:

    Input 784 --> BinaryLinear(784, 256) + ReLU
               --> BinaryLinear(256, 128) + ReLU
               --> BinaryLinear(128, 10)  --> logits

Two training changes, and nothing else (no batch norm, same Adam,
same learning rate, same 100 epochs):

    1. Forward weight = sign(latent) * SCALE / sqrt(fan_in)
       SCALE = 1 is full Xavier scaling and reaches the mid-90s.
       SCALE = 0.25 is intentionally milder.
    2. Latent weights are clipped to [-1, +1] after every step,
       so the straight-through estimator does not freeze.

Saved as:
{PTH_DIR}{date_name}_BNN.pth

'''



# ============================================
'''                   Imports              '''
# ============================================
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from paths import DATA_DIR, PTH_DIR




# ============================================
'''                Variables              '''
# ============================================
SEED = 42  # change this to try other runs; keep fixed for identical results
batch_size = 16
learning_rate = 1e-4

num_epochs = 100
# 1.0 reaches the mid-90s. 0.25 is a partial scale: better than the
# unscaled 02b run, short of a fully scaled binary network.
SCALE = 0.25
SAVE_FIGS = False


date_name = '28_oct_2026'


# ============================================
# 0) Reproducibility — fix all random sources
# ============================================

def set_seed(seed: int = SEED) -> None:
    """Set seeds so weight init, shuffling, and training are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(SEED)
print(f"Random seed set to {SEED} (reproducible run)")
print(f"Binary weight scale = {SCALE} / sqrt(fan_in)")


# ============================================
# 1) Transform MNIST
# ============================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1))
])

# ============================================
# 2) Load MNIST
# ============================================
train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    transform=transform,
    download=True
)


train_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=train_generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================
# 3) Binary Weight Function
# ============================================
class BinaryWeightFunction(torch.autograd.Function):
    """Forward: sign(w) in {-1, +1}. Backward: clipped STE in |w| <= 1."""

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.where(
            weight >= 0,
            torch.ones_like(weight),
            -torch.ones_like(weight)
        )

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_weight = grad_output.clone()
        grad_weight = grad_weight * (weight.abs() <= 1.0).float()
        return grad_weight


# ============================================
# 4) Binary Linear Layer
# ============================================
class BinaryLinear(nn.Linear):
    def binary_weight(self):
        scale = SCALE / math.sqrt(self.in_features)
        return BinaryWeightFunction.apply(self.weight) * scale

    def forward(self, input):
        return F.linear(input, self.binary_weight(), self.bias)


def clip_latent_weights(net):
    """Keep latent weights inside the region where the STE gradient is nonzero."""
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, BinaryLinear):
                module.weight.clamp_(-1.0, 1.0)


# ============================================
# 5) Binary MNIST Network
# ============================================
class BinaryMNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = BinaryLinear(784, 256)
        self.fc2 = BinaryLinear(256, 128)
        self.fc3 = BinaryLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ============================================
# 6) Model, Loss, Optimizer
# ============================================
model = BinaryMNIST()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================
# 7) Training
# ============================================

train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(outputs).any():
            print("NaN in outputs")
            break

        if torch.isnan(loss):
            print("NaN in loss")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_latent_weights(model)

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# ============================================
# 8) Evaluation
# ============================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


#------- SAVE --------

checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'test_accuracy': accuracy,
    'scale': SCALE,
}

checkpoint_path = PTH_DIR / f"{date_name}_BNN.pth"
torch.save(checkpoint, checkpoint_path)
print(f"Saved {checkpoint_path}")


# ============================================
# 9) Plot training loss
# ============================================
epochs = np.arange(1, num_epochs + 1)

plt.figure(figsize=(9, 5.5), dpi=120)
plt.plot(epochs, train_losses, color="#2563eb", linewidth=2.5)
plt.title(f"BNN training loss (scale = {SCALE})")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.xlim(1, num_epochs)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
