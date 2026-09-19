import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_loaders(batch_size=64, data_dir="./data_cifar"):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  train_dataset = datasets.CIFAR10(
      root=data_dir, train=True, download=True, transform=transform
  )
  test_dataset = datasets.CIFAR10(
      root=data_dir, train=False, download=True, transform=transform
  )
  return (
      DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
      DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
  )


class CIFARCNN(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32x32 -> 16x16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16x16 -> 8x8
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )

  def forward(self, x):
    return self.classifier(self.features(x))


def train_cifar():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Usando dispositivo: {device}")

  train_loader, test_loader = get_cifar_loaders(batch_size=64)
  model = CIFARCNN().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(
        f"Época {epoch+1}/5 | Pérdida train:"
        f" {running_loss/len(train_loader):.4f} | Precisión test: {acc:.2f}%"
    )

  os.makedirs("./checkpoints", exist_ok=True)
  torch.save(model.state_dict(), "./checkpoints/cifar_cnn.pth")
  print("✅ ¡Modelo CIFAR-10 guardado en ./checkpoints/cifar_cnn.pth!")


if __name__ == "__main__":
  train_cifar()