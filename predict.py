import torch
from torchvision import datasets, transforms

# Reutilizamos la misma arquitectura SimpleCNN
import torch.nn as nn

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

def predict_sample(index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    
    image, label = test_dataset[index]
    image_batch = image.unsqueeze(0).to(device) # (1, 1, 28, 28)
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("./checkpoints/mnist_cnn.pth"))
    model.eval()
    
    with torch.no_grad():
        output = model(image_batch)
        pred = output.argmax(dim=1).item()
        
    print(f"🖼️ Muestra de test #{index}")
    print(f"Etiqueta real     : {label}")
    print(f"Predicción modelo : {pred}")
    print("¡Acertó!" if label == pred else "❌ Falló")

if __name__ == "__main__":
    predict_sample(index=0)