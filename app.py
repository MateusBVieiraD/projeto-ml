import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Definir a CNN (precisa ser igual ao modelo treinado)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  
        return x

# Carregar o modelo treinado
modelo = CNN().to(device)
modelo.load_state_dict(torch.load("modelo_cachorros_gatos.pth", map_location= device))
modelo.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3,-1,-1) if x.shape[0] == 1 else x,),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

st.title("Classificador de Cachorros e Gatos 🐶🐱:")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

    image = transform(image).unsqueeze(0).to(device)  # Adiciona batch dimension

    with torch.no_grad():
        output = modelo(image)
        probabilities = F.softmax(output, dim = 1)
        confidence, predicted = torch.max(probabilities , 1)
        classes = ["Gato", "Cachorro"]

        threshold = 0.6

        if confidence.item() < threshold:
            st.write('🔍 A imagem pode não ser um gato nem um cachorro. 🔍')
        st.write(f"**Classe prevista:** {classes[predicted.item()]}")
