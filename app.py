import streamlit as st
import cv2
import numpy as np
import pygame
from pygame import mixer

# Configuración de Streamlit
st.title("Detector de movimiento con Streamlit")

# Configuración de Pygame para reproducir sonidos
pygame.init()
mixer.init()

# Carga el sonido
mixer.music.load("alarm-door-chime.wav")

# Función para ejecutar el detector de movimiento
def run_motion_detector():
    cap = cv2.VideoCapture(0)

    # Configurando la resolución del video
    cap.set(3, 640)  # Ancho del frame
    cap.set(4, 480)  # Altura del frame

    # Crear un contenedor para mostrar la imagen
    image_container = st.empty()

    while True:
        _, cam = cap.read()
        cam_gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
        
        # Configuración del detector de movimiento
        blur = cv2.GaussianBlur(cam_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 4000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Reproduce el sonido
            mixer.music.play()

        # Muestra el fotograma actual en el contenedor
        image_container.image(cam, channels="BGR", caption="Fotograma actual")

if __name__ == "__main__":
    run_motion_detector()
