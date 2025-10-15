# main.py
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel, Scale, HORIZONTAL, Label
from PIL import Image, ImageTk
import numpy as np
import cv2

# Asegúrate de que los otros archivos .py estén en la misma carpeta
from image_utils import load_image_from_dialog
from color_operations import convert_to_grayscale
from histogram_operations import display_histogram
from binarization import binarize_otsu, binarize_with_threshold
from arithmetic_operations import add_images, subtract_images, multiply_images, invert_image
from logical_operations import xor_images, and_images, or_images, not_image

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Principal")

        self.image1 = None
        self.image2 = None
        self.original_image1 = None

        # Referencias a las ventanas
        self.win_img1 = None
        self.win_img2 = None
        self.win_result = None

        # --- Interfaz de Controles ---
        control_frame = tk.Frame(root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Carga de imágenes
        tk.Button(control_frame, text="Cargar Imagen 1", command=self.load_image1).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Cargar Imagen 2", command=self.load_image2).pack(fill=tk.X, pady=2)

        # Procesamiento
        Label(control_frame, text="--- Procesamiento ---").pack(fill=tk.X, pady=(10,2))
        tk.Button(control_frame, text="Escala de Grises", command=self.apply_grayscale).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Binarizar (Otsu)", command=self.apply_otsu).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Binarizar Manual", command=self.manual_binarization_ui).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Invertir (Negativo)", command=self.apply_invert).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Mostrar Histograma", command=self.show_histogram).pack(fill=tk.X, pady=2)
        
        # Operaciones Aritméticas
        Label(control_frame, text="--- Aritméticas ---").pack(fill=tk.X, pady=(10,2))
        tk.Button(control_frame, text="Sumar", command=lambda: self.operand_choice_ui('add')).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Restar", command=lambda: self.operand_choice_ui('sub')).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="Multiplicar", command=lambda: self.operand_choice_ui('mul')).pack(fill=tk.X, pady=2)

        # Operaciones Lógicas
        Label(control_frame, text="--- Lógicas (Binarias) ---").pack(fill=tk.X, pady=(10,2))
        tk.Button(control_frame, text="XOR", command=self.apply_xor).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="AND", command=self.apply_and).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="OR", command=self.apply_or).pack(fill=tk.X, pady=2)
        tk.Button(control_frame, text="NOT", command=self.apply_not).pack(fill=tk.X, pady=2)
        
        # Deshacer
        Label(control_frame, text="--------------------").pack(fill=tk.X, pady=(10,2))
        tk.Button(control_frame, text="Deshacer", bg="orange", command=self.undo).pack(fill=tk.X, pady=5)

    def _create_or_update_window(self, window_ref, title, image):
        # Si la ventana no existe o fue cerrada, la crea de nuevo
        if not window_ref or not window_ref.winfo_exists():
            window_ref = Toplevel(self.root)
            window_ref.title(title)
            label = Label(window_ref)
            label.pack()
        
        # Actualiza la imagen
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label = window_ref.winfo_children()[0]
        label.config(image=img_tk)
        label.image = img_tk
        window_ref.title(title)
        
        return window_ref

    def load_image1(self):
        img = load_image_from_dialog()
        if img is not None:
            self.image1 = img
            self.original_image1 = img.copy()
            self.win_img1 = self._create_or_update_window(self.win_img1, "Imagen 1 (Original)", self.image1)
            # Evita que se cierre la ventana principal
            self.win_img1.protocol("WM_DELETE_WINDOW", lambda: messagebox.showinfo("Acción no permitida", "La ventana de la imagen original no se puede cerrar."))

    def load_image2(self):
        img = load_image_from_dialog()
        if img is not None:
            self.image2 = img
            self.win_img2 = self._create_or_update_window(self.win_img2, "Imagen 2", self.image2)

    def show_result(self, image, title):
        self.win_result = self._create_or_update_window(self.win_result, title, image)

    def apply_grayscale(self):
        if self.image1 is not None:
            gray_img = convert_to_grayscale(self.image1)
            self.show_result(gray_img, "Resultado: Escala de Grises")
        else: messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")

    def apply_otsu(self):
        if self.image1 is not None:
            gray_img = convert_to_grayscale(self.image1)
            bin_img = binarize_otsu(gray_img)
            self.show_result(bin_img, "Resultado: Binarización Otsu")
        else: messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        
    def manual_binarization_ui(self):
        if self.image1 is None:
            messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
            return

        gray_img = convert_to_grayscale(self.image1)
        
        # Ventana para el slider
        slider_window = Toplevel(self.root)
        slider_window.title("Ajuste de Umbral")
        
        def update_image(val):
            threshold = int(val)
            bin_img = binarize_with_threshold(gray_img, threshold)
            self.show_result(bin_img, f"Resultado: Umbral {threshold}")

        scale = Scale(slider_window, from_=0, to=255, orient=HORIZONTAL, label="Umbral", command=update_image, length=300)
        scale.set(127)
        scale.pack(pady=10, padx=10)
        update_image(127)

    def operand_choice_ui(self, operation_name):
        if self.image1 is None:
            messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
            return

        choice_win = Toplevel(self.root)
        choice_win.title("Elegir Operando")
        
        op_map = {'add': add_images, 'sub': subtract_images, 'mul': multiply_images}
        operation_func = op_map[operation_name]

        def on_image_choice():
            if self.image2 is None:
                messagebox.showwarning("Advertencia", "Cargue la Imagen 2 primero.")
                return
            result = operation_func(self.image1, self.image2)
            self.show_result(result, f"Resultado: {operation_name.capitalize()}")
            choice_win.destroy()

        def on_scalar_choice():
            scalar = simpledialog.askinteger("Input", "Ingrese el valor escalar:", parent=choice_win)
            if scalar is not None:
                result = operation_func(self.image1, scalar)
                self.show_result(result, f"Resultado: {operation_name.capitalize()} (escalar {scalar})")
            choice_win.destroy()

        tk.Button(choice_win, text="Usar Imagen 2", command=on_image_choice).pack(pady=5, padx=20)
        tk.Button(choice_win, text="Usar Escalar", command=on_scalar_choice).pack(pady=5, padx=20)

    def _execute_logical(self, op_func, op_name):
        if self.image1 is None or self.image2 is None:
            messagebox.showwarning("Advertencia", "Cargue ambas imágenes para esta operación.")
            return
        bin1 = binarize_otsu(convert_to_grayscale(self.image1))
        bin2 = binarize_otsu(convert_to_grayscale(self.image2))
        result = op_func(bin1, bin2)
        self.show_result(result, f"Resultado: {op_name}")

    def apply_xor(self): self._execute_logical(xor_images, "XOR")
    def apply_and(self): self._execute_logical(and_images, "AND")
    def apply_or(self): self._execute_logical(or_images, "OR")

    def apply_not(self):
        if self.image1 is not None:
            bin1 = binarize_otsu(convert_to_grayscale(self.image1))
            result = not_image(bin1)
            self.show_result(result, "Resultado: NOT")
        else: messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")

    def apply_invert(self):
        if self.image1 is not None:
            inverted = invert_image(self.image1)
            self.show_result(inverted, "Resultado: Invertido (Negativo)")
        else: messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        
    def show_histogram(self):
        if self.image1 is not None:
            # Decide qué imagen usar para el histograma
            img_to_process = self.win_result.winfo_children()[0].image if (self.win_result and self.win_result.winfo_exists()) else self.image1
            display_histogram(img_to_process)
        else:
            messagebox.showwarning("Advertencia", "No hay imagen para mostrar su histograma.")

    def undo(self):
        if self.original_image1 is not None:
            self.image1 = self.original_image1.copy()
            # Cierra ventanas de resultado e imagen 2 si existen
            if self.win_result and self.win_result.winfo_exists(): self.win_result.destroy()
            if self.win_img2 and self.win_img2.winfo_exists(): self.win_img2.destroy()
            self.image2 = None
            self.win_img2 = None
            
            # Actualiza la ventana original
            self.win_img1 = self._create_or_update_window(self.win_img1, "Imagen 1 (Restaurada)", self.image1)
            messagebox.showinfo("Información", "Cambios deshechos.")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen original para restaurar.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
