import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def evaluate_full_metrics():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform_test = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  test_dataset = datasets.CIFAR10(
      root="./data_cifar",
      train=False,
      download=False,
      transform=transform_test,
  )
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  model = models.resnet18(weights=None)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 10)
  model.load_state_dict(
      torch.load("./checkpoints/resnet18_cifar.pth", weights_only=True)
  )
  model.to(device)
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
  print("\n=== CLASSIFICATION REPORT (ResNet18 - 94.25%) ===")
  print(
      classification_report(
          all_targets, all_preds, target_names=classes, digits=4
      )
  )


if __name__ == "__main__":
  evaluate_full_metrics()