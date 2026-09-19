import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_cifar_transfer_loaders(batch_size=64):
  # ResNet espera imágenes de 224x224 y normalización de ImageNet
  transform_train = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(
      ),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  transform_test = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  train_dataset = datasets.CIFAR10(
      root="./data_cifar", train=True, download=True, transform=transform_train
  )
  test_dataset = datasets.CIFAR10(
      root="./data_cifar", train=False, download=False, transform=transform_test
  )

  return (
      DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
      DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
  )


def get_resnet18_model(num_classes=10):
  # Cargar pesos preentrenados modernos
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  # Congelar capas base (opcional, o fine-tuning completo directito)
  # Para este salto rápido, reentrenamos la capa final adaptándola a 10 clases:
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)
  return model


def train_transfer():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Usando dispositivo: {device}")

  train_loader, test_loader = get_cifar_transfer_loaders(batch_size=64)
  model = get_resnet18_model(num_classes=10).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)  # LR más pequeño para transfer learning

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
  torch.save(model.state_dict(), "./checkpoints/resnet18_cifar.pth")
  print("✅ ¡Modelo ResNet18 guardado en ./checkpoints/resnet18_cifar.pth!")


if __name__ == "__main__":
  train_transfer()