import streamlit as st
import torch
from generator import Generator  # Asegúrate de que este es el mismo que entrenaste
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

# Interfaz
st.title("Generador de Dígitos Manuscritos (MNIST)")
digit = st.selectbox("Selecciona un dígito (0-9):", list(range(10)))

if st.button("Generar Imágenes"):
    z = torch.randn(5, 100).to(device)
    labels = torch.tensor([digit] * 5).to(device)
    with torch.no_grad():
        generated_imgs = model(z, labels).cpu()

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(generated_imgs[i][0], cmap="gray")
        axs[i].axis('off')
    st.pyplot(fig)
