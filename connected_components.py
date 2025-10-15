# connected_components.py
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def label_and_show_components(binary_image, title="Etiquetado de Componentes Conexas"):
    """
    Aplica el etiquetado de componentes conexas con vecindad 4 y 8
    y muestra los resultados usando Matplotlib en una nueva ventana de Tkinter.
    """
    if binary_image is None or len(binary_image.shape) != 2:
        print("Error: La imagen de entrada no es una imagen binaria válida.")
        return

    # Paso 1: Definir vecindades (estructuras de conectividad)
    # Vecindad 4 (Ortogonal)
    vecindad_4 = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=int)
    
    # Vecindad 8 (Incluye diagonales)
    vecindad_8 = np.ones((3, 3), dtype=int)

    # Paso 2: Aplicar el Etiquetado
    etiquetas_4, num_objetos_4 = ndimage.label(binary_image, structure=vecindad_4)
    etiquetas_8, num_objetos_8 = ndimage.label(binary_image, structure=vecindad_8)

    # Paso 3: Crear y mostrar la ventana con Matplotlib
    
    # Crear la ventana de Tkinter
    plot_window = tk.Toplevel()
    plot_window.title(title)
    
    # Crea la figura de 1 fila, 3 columnas
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 1. Imagen binaria original
    axes[0].imshow(binary_image, cmap='gray')
    axes[0].set_title("Imagen Binaria Original")
    axes[0].axis('off')
    
    # 2. Etiquetado con vecindad 4
    # Usamos cmap='nipy_spectral' para que cada etiqueta tenga un color diferente
    axes[1].imshow(etiquetas_4, cmap='nipy_spectral')
    axes[1].set_title(f"Vecindad 4: {num_objetos_4} Objetos")
    axes[1].axis('off')
    
    # 3. Etiquetado con vecindad 8
    axes[2].imshow(etiquetas_8, cmap='nipy_spectral')
    axes[2].set_title(f"Vecindad 8: {num_objetos_8} Objetos")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Integrar el gráfico de Matplotlib en la ventana de Tkinter
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)