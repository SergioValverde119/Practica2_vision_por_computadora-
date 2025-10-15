# histogram_operations.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def calculate_histogram(image):
    """Calcula el histograma de una imagen (grises o un solo canal)."""
    hist = np.zeros(256)
    for pixel_value in range(256):
        hist[pixel_value] = np.sum(image == pixel_value)
    return hist

def display_histogram(image):
    """Muestra el histograma de la imagen en una nueva ventana de Tkinter."""
    hist_window = tk.Toplevel()
    hist_window.title("Histograma")
    
    fig, ax = plt.subplots()

    if len(image.shape) == 2: # Grayscale or Binary
        hist = calculate_histogram(image)
        ax.plot(hist, color='black')
        ax.set_title("Histograma de Intensidad")
    else: # RGB
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = calculate_histogram(image[:,:,i])
            ax.plot(hist, color=color)
        ax.set_title("Histograma por Canal (R-G-B)")
        ax.legend(['Azul', 'Verde', 'Rojo'])

    ax.set_xlim([0, 256])
    
    canvas = FigureCanvasTkAgg(fig, master=hist_window)
    canvas.draw()
    canvas.get_tk_widget().pack()
