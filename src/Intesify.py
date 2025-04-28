import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np


class PixelIntensityApp:
    MAX_DISPLAY_WIDTH = 800
    MAX_DISPLAY_HEIGHT = 600

    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор пикселей")

        # Переменные для данных изображения
        self.pixels = None
        self.photo_image = None
        self.image_size = (0, 0)
        self.scale_factor = 1.0
        self.canvas = None
        self.marker = None

        # Настройка сетки окна
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(1, weight=1)

        # Создание элементов GUI
        self.create_widgets()

    def create_widgets(self):
        # Кнопка загрузки изображения
        self.load_btn = tk.Button(self.root, text="Загрузить изображение", command=self.load_image)
        self.load_btn.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        # Canvas для отображения изображения и маркера
        self.canvas = tk.Canvas(self.root, width=self.MAX_DISPLAY_WIDTH, height=self.MAX_DISPLAY_HEIGHT)
        self.canvas.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        # Вертикальный слайдер справа
        self.y_slider = tk.Scale(self.root, from_=0, to=0, orient='vertical',
                                 resolution=1, command=self.update_intensity)
        self.y_slider.grid(row=1, column=1, padx=5, pady=5, sticky='ns')

        # Горизонтальный слайдер снизу
        self.x_slider = tk.Scale(self.root, from_=0, to=0, orient='horizontal',
                                 resolution=1, command=self.update_intensity)
        self.x_slider.grid(row=2, column=0, padx=5, pady=5, sticky='ew')

        # Метка для отображения интенсивности
        self.intensity_label = tk.Label(self.root, text="Интенсивность: ")
        self.intensity_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert('L')
            self.image_size = img.size

            # Очистка предыдущего изображения и маркера
            self.canvas.delete("all")
            self.marker = None

            self.pixels = np.array(img)
            img_rgb = img.convert('RGB')
            img_display = self.resize_image(img_rgb)

            # Центрирование изображения на Canvas
            self.photo_image = ImageTk.PhotoImage(img_display)
            img_width = img_display.width
            img_height = img_display.height

            # Координаты для центрирования изображения
            x_offset = (self.MAX_DISPLAY_WIDTH - img_width) // 2
            y_offset = (self.MAX_DISPLAY_HEIGHT - img_height) // 2

            self.canvas.create_image(
                x_offset, y_offset,
                anchor='nw', image=self.photo_image,
                tags="image"
            )

            # Создание маркера
            self.marker = self.canvas.create_oval(
                x_offset - 3, y_offset - 3,
                x_offset + 3, y_offset + 3,
                fill='red', state='hidden'
            )

            # Сохранение параметров отображения
            self.display_params = {
                'x_offset': x_offset,
                'y_offset': y_offset,
                'img_width': img_width,
                'img_height': img_height
            }

            # Настройка слайдеров
            self.x_slider.config(to=self.image_size[0] - 1)
            self.y_slider.config(to=self.image_size[1] - 1)
            self.x_slider.set(0)
            self.y_slider.set(0)

            if self.marker:
                self.canvas.itemconfig(self.marker, state='normal')

        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")

    def resize_image(self, img):
        original_width, original_height = img.size
        width_scale = self.MAX_DISPLAY_WIDTH / original_width
        height_scale = self.MAX_DISPLAY_HEIGHT / original_height
        self.scale_factor = min(width_scale, height_scale)

        if self.scale_factor < 1:
            new_size = (
                int(original_width * self.scale_factor),
                int(original_height * self.scale_factor)
            )
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def update_intensity(self, *args):
        if self.pixels is None or not self.marker:
            return

        x = int(self.x_slider.get())
        y = int(self.y_slider.get())

        # Преобразование координат с учетом масштаба и смещения
        display_x = x * self.scale_factor + self.display_params['x_offset']
        display_y = y * self.scale_factor + self.display_params['y_offset']

        # Обновление позиции маркера
        self.canvas.coords(
            self.marker,
            display_x - 3, display_y - 3,
            display_x + 3, display_y + 3
        )

        intensity = self.pixels[y, x]
        self.intensity_label.config(
            text=f"Интенсивность: {intensity} (X: {x}, Y: {y})"
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x700")
    app = PixelIntensityApp(root)
    root.mainloop()