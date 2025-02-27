import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

modelo = keras.models.load_model('modelo.h5')

class_names = ['Cat','Dog']

# limiar de confiança

confidence_threshold = 0.7

st.title('Classificação de Animais: 🐶🐱')

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img.convert('RGB')
    img = img.resize((64,64))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = modelo.predict(img_array)[0][0]

    # limiar pra rejeição
    if prediction  < (1 - confidence_threshold):
        st.warning("A imagem se parece mais com um 🐱 Gato.")
    elif prediction > confidence_threshold:  
        st.warning('A imagem se parece mais com um 🐶 Cachorro.')
    else:
        st.error("Não tenho certeza! Talvez seja outro animal. 🤔")