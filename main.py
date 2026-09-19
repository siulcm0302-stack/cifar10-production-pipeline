import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1. Dataset y DataLoader
def get_mnist_loaders(batch_size=64, data_dir="./data"):
  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
  )
  train_dataset = datasets.MNIST(
      root=data_dir, train=True, download=True, transform=transform
  )
  test_dataset = datasets.MNIST(
      root=data_dir, train=False, download=True, transform=transform
  )
  return (
      DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
      DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
  )


# 2. Modelo CNN
class SimpleCNN(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )

  def forward(self, x):
    return self.classifier(self.features(x))


# 3. Entrenamiento
def train_model(
    model, train_loader, test_loader, epochs=3, lr=0.001, device="cpu"
):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  model.to(device)

  for epoch in range(epochs):
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
        f"Época {epoch+1:2d}/{epochs:2d} | Pérdida train:"
        f" {running_loss/len(train_loader):.4f} | Precisión test: {acc:.2f}%"
    )


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Usando dispositivo: {device}")
  if device.type == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

  train_loader, test_loader = get_mnist_loaders(batch_size=64)
  model = SimpleCNN()
  train_model(
      model, train_loader, test_loader, epochs=3, lr=0.001, device=device
  )

  # Guardar pesos
  os.makedirs("./checkpoints", exist_ok=True)
  torch.save(model.state_dict(), "./checkpoints/mnist_cnn.pth")
  print("✅ ¡Modelo guardado en ./checkpoints/mnist_cnn.pth!")


if __name__ == "__main__":
  main()