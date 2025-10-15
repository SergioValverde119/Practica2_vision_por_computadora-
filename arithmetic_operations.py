# arithmetic_operations.py
import numpy as np
import cv2

def add_images(image1, operand):
    """Suma una imagen y otra imagen o un escalar."""
    if isinstance(operand, np.ndarray): # Suma de dos imágenes
        # Asegurarse de que las imágenes tengan el mismo tamaño
        operand = cv2.resize(operand, (image1.shape[1], image1.shape[0]))
    result = cv2.add(image1, operand)
    return result

def subtract_images(image1, operand):
    """Resta una imagen y otra imagen o un escalar."""
    if isinstance(operand, np.ndarray):
        operand = cv2.resize(operand, (image1.shape[1], image1.shape[0]))
    result = cv2.subtract(image1, operand)
    return result

def multiply_images(image1, operand):
    """Multiplica una imagen y otra imagen o un escalar."""
    if isinstance(operand, np.ndarray):
        operand = cv2.resize(operand, (image1.shape[1], image1.shape[0]))
        # La multiplicación por píxel se realiza con cv2.multiply
        result = cv2.multiply(image1, operand, scale=1/255) # Escalar para mantener en rango 0-255
    else: # Multiplicación por escalar
        result = cv2.multiply(image1, np.array([operand]))
    return np.clip(result, 0, 255).astype(np.uint8)

def invert_image(image):
    """Invierte una imagen usando operaciones aritméticas (255 - píxel)."""
    return 255 - image
