# logical_operations.py
import numpy as np
import cv2

def xor_images(image1, image2):
    """Realiza la operación XOR entre dos imágenes binarias."""
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return cv2.bitwise_xor(image1, image2_resized)

def and_images(image1, image2):
    """Realiza la operación AND entre dos imágenes binarias."""
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return cv2.bitwise_and(image1, image2_resized)

def or_images(image1, image2):
    """Realiza la operación OR entre dos imágenes binarias."""
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    return cv2.bitwise_or(image1, image2_resized)

def not_image(image):
    """Realiza la operación NOT en una imagen binaria."""
    return cv2.bitwise_not(image)
