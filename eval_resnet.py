import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_cifar_transfer_test_loader(batch_size=64):
  transform_test = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])
  test_dataset = datasets.CIFAR10(
      root="./data_cifar", train=False, download=False, transform=transform_test
  )
  return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_resnet18_model(num_classes=10):
  model = models.resnet18(weights=None)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)
  return model


def plot_resnet_confusion_matrix():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_loader = get_cifar_transfer_test_loader()

  model = get_resnet18_model(num_classes=10).to(device)
  model.load_state_dict(
      torch.load("./checkpoints/resnet18_cifar.pth", weights_only=True)
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
  plt.title("Matriz de Confusión - ResNet18 CIFAR-10 (RTX 4050)", fontsize=14)
  plt.tight_layout()

  os.makedirs("./checkpoints", exist_ok=True)
  plt.savefig("./checkpoints/resnet18_confusion_matrix.png")
  print("✅ Matriz guardada en ./checkpoints/resnet18_confusion_matrix.png")
  plt.show()


if __name__ == "__main__":
  plot_resnet_confusion_matrix()