import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar forward/backward
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # Global Average Pooling sobre batch (0), alto (2) y ancho (3)
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        
        # Ponderar canales de activación
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, class_idx

def get_resnet18_model(num_classes=10):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def run_gradcam(sample_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.CIFAR10(root="./data_cifar", train=False, download=False, transform=transform_test)
    raw_dataset = datasets.CIFAR10(root="./data_cifar", train=False, download=False, transform=transforms.Resize((224, 224)))
    
    image_tensor, label = test_dataset[sample_index]
    raw_image, _ = raw_dataset[sample_index]
    
    model = get_resnet18_model(num_classes=10).to(device)
    model.load_state_dict(torch.load("./checkpoints/resnet18_cifar.pth", weights_only=True))
    
    # Capa objetivo corregida para ResNet18 (último bloque de layer4, su segunda convolución)
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)
    
    input_tensor = image_tensor.unsqueeze(0).to(device)
    heatmap, pred_idx = grad_cam(input_tensor)
    
    classes = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
    
    # Redimensionar heatmap al tamaño de la imagen (224x224)
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224), Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_pil) / 255.0
    
    raw_np = np.array(raw_image) / 255.0
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(raw_np)
    ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
    ax.axis('off')
    plt.title(f"Real: {classes[label]} | Predicción: {classes[pred_idx]}", fontsize=12)
    plt.tight_layout()
    
    os.makedirs("./checkpoints", exist_ok=True)
    plt.savefig(f"./checkpoints/gradcam_sample_{sample_index}.png")
    print(f"✅ Grad-CAM guardado en ./checkpoints/gradcam_sample_{sample_index}.png")
    plt.show()

if __name__ == "__main__":
    run_gradcam(sample_index=0)