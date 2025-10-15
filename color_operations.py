# color_operations.py
import numpy as np

def separate_rgb(image):
    """
    Separa una imagen en sus canales R, G, B manualmente.
    Devuelve tres arrays de NumPy, uno para cada canal.
    """
    b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
    
    # Para visualización, creamos imágenes que solo contienen ese canal
    blue_channel_img = np.zeros_like(image)
    blue_channel_img[:,:,0] = b
    
    green_channel_img = np.zeros_like(image)
    green_channel_img[:,:,1] = g
    
    red_channel_img = np.zeros_like(image)
    red_channel_img[:,:,2] = r
    
    return red_channel_img, green_channel_img, blue_channel_img

def convert_to_grayscale(image):
    """
    Convierte una imagen a escala de grises manualmente usando la fórmula de luminosidad.
    Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 2: # La imagen ya está en escala de grises
        return image
        
    b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
    
    # Aplicar la fórmula de luminosidad
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Convertir a tipo de dato de 8 bits sin signo
    return gray_image.astype(np.uint8)
