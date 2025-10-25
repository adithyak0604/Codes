# LeNet Implementation in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# Define LeNet-5 Architecture
# ------------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # For grayscale images (MNIST) / For Cifar, (3, 6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjust depending on input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # Conv1 + ReLU
        x = F.max_pool2d(x, 2)      # Pooling
        x = F.relu(self.conv2(x))   # Conv2 + ReLU
        x = F.max_pool2d(x, 2)      # Pooling
        x = torch.flatten(x, 1)     # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # Output logits
        return x

# ------------------------------
# Training Setup
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
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

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
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({100. * correct / len(test_loader.dataset):.2f}%)\n")

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST dataset (handwritten digits)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),   # LeNet expects 32x32 input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  #For Cifar, .Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)  #.CIFAR10
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)  #.CIFAR10

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and test
    for epoch in range(1, 6):  # Train for 5 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
