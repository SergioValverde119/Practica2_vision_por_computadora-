# image_utils.py
from tkinter import filedialog
import cv2

def load_image_from_dialog():
    """Abre un diálogo para seleccionar un archivo de imagen y la carga con OpenCV."""
    file_path = filedialog.askopenfilename()
    if file_path:
        # Cargar la imagen en formato BGR por defecto con OpenCV
        image = cv2.imread(file_path)
        if image is not None:
            return image
    return None
