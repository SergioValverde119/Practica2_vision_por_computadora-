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

# --- Constantes para el tamaño de visualización ---
MAX_WIDTH = 900
MAX_HEIGHT = 700

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Principal")

        self.image1 = None
        self.image2 = None
        self.original_image1 = None
        self.result_image = None # Para guardar el resultado

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
        
        # Deshacer y Guardar
        Label(control_frame, text="--------------------").pack(fill=tk.X, pady=(10,2))
        tk.Button(control_frame, text="Deshacer", bg="orange", command=self.undo).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="Guardar Resultado", bg="lightgreen", command=self.save_result).pack(fill=tk.X, pady=5)

    def _create_or_update_window(self, window_ref, title, image):
        # Lógica para redimensionar la imagen si es demasiado grande para la pantalla
        display_image = image.copy()
        h, w = display_image.shape[:2]

        if w > MAX_WIDTH or h > MAX_HEIGHT:
            ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            display_image = cv2.resize(display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if not window_ref or not window_ref.winfo_exists():
            window_ref = Toplevel(self.root)
            window_ref.transient(self.root)
            label = Label(window_ref)
            label.pack()
        
        img_pil = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) if len(display_image.shape) == 3 else display_image)
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
            self.win_img1.protocol("WM_DELETE_WINDOW", lambda: messagebox.showinfo("Acción no permitida", "La ventana de la imagen original no se puede cerrar."))

    def load_image2(self):
        img = load_image_from_dialog()
        if img is not None:
            self.image2 = img
            self.win_img2 = self._create_or_update_window(self.win_img2, "Imagen 2", self.image2)

    def show_result(self, image, title):
        self.result_image = image # Guardar referencia para la función de guardado
        self.win_result = self._create_or_update_window(self.win_result, title, image)

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Advertencia", "No hay ninguna imagen de resultado para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Éxito", f"Imagen guardada en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen.\nError: {e}")

    def apply_grayscale(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        gray_img = convert_to_grayscale(self.image1)
        self.show_result(gray_img, "Resultado: Escala de Grises")

    def apply_otsu(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        gray_img = convert_to_grayscale(self.image1)
        bin_img = binarize_otsu(gray_img)
        self.show_result(bin_img, "Resultado: Binarización Otsu")

    def manual_binarization_ui(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        gray_img = convert_to_grayscale(self.image1)
        
        slider_window = Toplevel(self.root)
        slider_window.title("Ajuste de Umbral")
        slider_window.transient(self.root)
        slider_window.grab_set()
        
        def update_image(val):
            threshold = int(val)
            bin_img = binarize_with_threshold(gray_img, threshold)
            self.show_result(bin_img, f"Resultado: Umbral {threshold}")

        scale = Scale(slider_window, from_=0, to=255, orient=HORIZONTAL, label="Umbral", command=update_image, length=300)
        scale.set(127)
        scale.pack(pady=10, padx=10)
        update_image(127)

    def operand_choice_ui(self, operation_name):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        
        choice_win = Toplevel(self.root)
        choice_win.title("Elegir Operando")
        choice_win.transient(self.root)
        choice_win.grab_set()
        
        op_map = {'add': add_images, 'sub': subtract_images, 'mul': multiply_images}
        op_func = op_map[operation_name]

        def on_image_choice():
            choice_win.destroy()
            if self.image2 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 2 primero.")
            result = op_func(self.image1, self.image2)
            self.show_result(result, f"Resultado: {operation_name.capitalize()}")

        def on_scalar_choice():
            scalar = simpledialog.askinteger("Input", "Ingrese el valor escalar:", parent=choice_win)
            choice_win.destroy()
            if scalar is not None:
                result = op_func(self.image1, scalar)
                self.show_result(result, f"Resultado: {operation_name.capitalize()} (escalar {scalar})")

        tk.Button(choice_win, text="Usar Imagen 2", command=on_image_choice).pack(pady=5, padx=20, fill=tk.X)
        tk.Button(choice_win, text="Usar Escalar", command=on_scalar_choice).pack(pady=5, padx=20, fill=tk.X)

    def _execute_logical(self, op_func, op_name):
        if self.image1 is None or self.image2 is None: return messagebox.showwarning("Advertencia", "Cargue ambas imágenes.")
        bin1 = binarize_otsu(convert_to_grayscale(self.image1))
        bin2 = binarize_otsu(convert_to_grayscale(self.image2))
        result = op_func(bin1, bin2)
        self.show_result(result, f"Resultado: {op_name}")

    def apply_xor(self): self._execute_logical(xor_images, "XOR")
    def apply_and(self): self._execute_logical(and_images, "AND")
    def apply_or(self): self._execute_logical(or_images, "OR")

    def apply_not(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        bin1 = binarize_otsu(convert_to_grayscale(self.image1))
        self.show_result(not_image(bin1), "Resultado: NOT")

    def apply_invert(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "Cargue la Imagen 1 primero.")
        self.show_result(invert_image(self.image1), "Resultado: Invertido (Negativo)")

    def show_histogram(self):
        if self.image1 is None: return messagebox.showwarning("Advertencia", "No hay imagen para mostrar su histograma.")
        display_histogram(self.image1)

    def undo(self):
        if self.original_image1 is not None:
            self.image1 = self.original_image1.copy()
            self.result_image = None
            if self.win_result and self.win_result.winfo_exists(): self.win_result.destroy()
            if self.win_img2 and self.win_img2.winfo_exists(): self.win_img2.destroy()
            self.image2 = None; self.win_img2 = None
            
            self.win_img1 = self._create_or_update_window(self.win_img1, "Imagen 1 (Restaurada)", self.image1)
            messagebox.showinfo("Información", "Cambios deshechos.")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen original para restaurar.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

