# Basic CNN for MNIST Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# Define a Simple CNN
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 input channel, 8 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 16 filters
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Fully connected layer
        self.fc2 = nn.Linear(64, 10)          # 10 classes (digits)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Conv1 + ReLU
        x = F.max_pool2d(x, 2)         # Pooling (reduces size)
        x = F.relu(self.conv2(x))      # Conv2 + ReLU
        x = F.max_pool2d(x, 2)         # Pooling
        x = torch.flatten(x, 1)        # Flatten for FC
        x = F.relu(self.fc1(x))        # Fully connected
        x = self.fc2(x)                # Output layer
        return x

# ------------------------------
# Training Function
# ------------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

# ------------------------------
# Testing Function
# ------------------------------
def test(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%\n")

# ------------------------------
# Main Function
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model, optimizer
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 3 epochs (fast for lab demo)
    for epoch in range(1, 4):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == "__main__":
    main()