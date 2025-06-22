import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define Generator
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        input = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(input)
        return img.view(img.size(0), *self.img_shape)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# Optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
loss_fn = nn.BCELoss()

# Training loop
epochs = 20
for epoch in range(epochs):
    for imgs, labels in dataloader:
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1).to(device)
        noise = torch.randn(batch_size, 100).to(device)
        labels = labels.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        gen_imgs = generator(noise, labels)
        validity = torch.ones(batch_size, 1).to(device)
        g_loss = loss_fn(validity, real)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs} | Generator Loss: {g_loss.item():.4f}")

torch.save(generator.state_dict(), "generator.pth")
