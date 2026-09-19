import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Arquitectura CIFARCNN
class CIFARCNN(nn.Module):

  def __init__(self, num_classes=10):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
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


def get_cifar_test_loader(batch_size=64):
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  test_dataset = datasets.CIFAR10(
      root="./data_cifar", train=False, download=False, transform=transform
  )
  return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def plot_confusion_matrix():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_loader = get_cifar_test_loader()

  model = CIFARCNN().to(device)
  model.load_state_dict(
      torch.load("./checkpoints/cifar_cnn.pth", weights_only=True)
  )
  model.eval()

  all_preds = []
  all_targets = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      outputs = model(images)
      preds = outputs.argmax(dim=1).cpu().numpy()
      all_preds.extend(preds)
      all_targets.extend(labels.numpy())

  classes = [
      "avión",
      "automóvil",
      "pájaro",
      "gato",
      "ciervo",
      "perro",
      "rana",
      "caballo",
      "barco",
      "camión",
  ]
  cm = confusion_matrix(all_targets, all_preds)

  fig, ax = plt.subplots(figsize=(10, 8))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format="d")
  plt.title("Matriz de Confusión - CIFAR-10 (RTX 4050)", fontsize=14)
  plt.tight_layout()

  os.makedirs("./checkpoints", exist_ok=True)
  plt.savefig("./checkpoints/cifar_confusion_matrix.png")
  print(
      "✅ Matriz guardada en ./checkpoints/cifar_confusion_matrix.png"
  )
  plt.show()


if __name__ == "__main__":
  plot_confusion_matrix()