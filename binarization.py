# binarization.py
import numpy as np

def binarize_otsu(gray_image):
    """
    Binariza una imagen en escala de grises usando el método de Otsu implementado manualmente.
    """
    # Calcular histograma
    hist = np.histogram(gray_image.ravel(), 256, [0,256])[0]
    
    total_pixels = gray_image.size
    
    current_max, threshold = 0, 0
    sum_total, sum_background = 0, 0
    weight_background, weight_foreground = 0, 0

    # Suma total de intensidades
    for i in range(256):
        sum_total += i * hist[i]

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += i * hist[i]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calcular la varianza entre clases
        variance_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)

        if variance_between > current_max:
            current_max = variance_between
            threshold = i

    # Aplicar el umbral encontrado
    binary_image = (gray_image > threshold).astype(np.uint8) * 255
    return binary_image

def binarize_with_threshold(gray_image, threshold):
    """Binariza una imagen usando un umbral dado."""
    return (gray_image > threshold).astype(np.uint8) * 255
