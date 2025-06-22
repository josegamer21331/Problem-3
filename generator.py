import torch
from torch import nn

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
