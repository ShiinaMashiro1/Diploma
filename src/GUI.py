import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from PIL import Image, ImageTk

from src.Grey_fade import grey_scale
from src.Wavelegth_calc import graphscalc
from src.detector import detect

# ========== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ==========
INPUT_DIR = "E:\CMU\Code\Sky_samples"
OUTPUT_PATH = "E:\CMU\Code\Samples_res\output_test.jpg"
MEDIAN_BLUR_SIZE = 19
BILATERAL_D = 2
BILATERAL_SIGMA_COLOR = 10
BILATERAL_SIGMA_SPACE = 10
CROP_WIDTH = 30
CROP_HEIGHT = 2000


# ==========================================

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработчик изображений неба")
        self.geometry("700x700")
        self.progress_lock = threading.Lock()
        self.create_widgets()

    def create_widgets(self):
        # Выбор директории с изображениями
        input_frame = ttk.Frame(self)
        input_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(input_frame, text="Папка с изображениями:").pack(side='left')
        self.input_dir = tk.StringVar(value=INPUT_DIR)
        ttk.Entry(input_frame, textvariable=self.input_dir, width=50).pack(side='left', expand=True, fill='x', padx=5)
        ttk.Button(input_frame, text="Обзор", command=self.select_input_dir).pack(side='left')

        # Выбор файла для сохранения
        output_frame = ttk.Frame(self)
        output_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(output_frame, text="Файл для сохранения:").pack(side='left')
        self.output_path = tk.StringVar(value=OUTPUT_PATH)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side='left', expand=True, fill='x',
                                                                              padx=5)
        ttk.Button(output_frame, text="Обзор", command=self.select_output_file).pack(side='left')

        # Параметры обработки
        params_frame = ttk.LabelFrame(self, text="Параметры обработки")
        params_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.create_parameter_row(params_frame, "Размер медианного фильтра (нечетный):",
                                  'median_blur', MEDIAN_BLUR_SIZE, 0)
        self.create_parameter_row(params_frame, "Диаметр билатерального фильтра:",
                                  'bilateral_d', BILATERAL_D, 1)
        self.create_parameter_row(params_frame, "Сигма цвета (билатеральный):",
                                  'sigma_color', BILATERAL_SIGMA_COLOR, 2)
        self.create_parameter_row(params_frame, "Сигма пространства (билатеральный):",
                                  'sigma_space', BILATERAL_SIGMA_SPACE, 3)
        self.create_parameter_row(params_frame, "Ширина области обрезки (X):",
                                  'crop_width', CROP_WIDTH, 4)
        self.create_parameter_row(params_frame, "Высота области обрезки (Y):",
                                  'crop_height', CROP_HEIGHT, 5)

        # Прогресс-бар
        self.progress = ttk.Progressbar(self, orient='horizontal', mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=5)

        # Кнопки управления
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.process_btn = ttk.Button(btn_frame, text="Начать обработку", command=self.start_processing)
        self.process_btn.pack(side='left', padx=5)
        self.preview_btn = ttk.Button(btn_frame, text="Полученное изображение",
                                      command=self.show_preview, state='disabled')
        self.preview_btn.pack(side='left', padx=5)

    def create_parameter_row(self, frame, label_text, var_name, default_value, row):
        var = tk.IntVar(value=default_value)
        setattr(self, var_name, var)
        ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(frame, textvariable=var, width=10).grid(row=row, column=1, sticky='w', padx=5, pady=2)

    def select_input_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)

    def select_output_file(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if path:
            self.output_path.set(path)

    def start_processing(self):
        try:
            params = {
                'input_dir': self.input_dir.get(),
                'output_path': self.output_path.get(),
                'median_blur': self.median_blur.get(),
                'bilateral_d': self.bilateral_d.get(),
                'sigma_color': self.sigma_color.get(),
                'sigma_space': self.sigma_space.get(),
                'crop_width': self.crop_width.get(),
                'crop_height': self.crop_height.get()
            }
        except tk.TclError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные параметры: {str(e)}")
            return

        self.process_btn.config(state='disabled')
        self.preview_btn.config(state='disabled')
        self.progress['value'] = 0
        threading.Thread(target=self.run_processing, kwargs=params, daemon=True).start()

    def update_progress(self, value):
        if isinstance(value, (int, float)):
            with self.progress_lock:
                self.progress['value'] = value
                self.update_idletasks()

    def run_processing(self, **kwargs):
        try:
            global INPUT_DIR, OUTPUT_PATH, MEDIAN_BLUR_SIZE, BILATERAL_D
            global BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE, CROP_WIDTH, CROP_HEIGHT

            # Обновляем глобальные переменные
            for key, value in kwargs.items():
                globals()[key.upper()] = value

            # Обработка изображений
            self.process_images_with_progress()

            # Дополнительные этапы обработки
            grey_scale()
            detect()


            self.after(0, lambda: messagebox.showinfo("Готово", "Обработка успешно завершена!"))
            self.after(0, lambda: self.preview_btn.config(state='normal'))

        except Exception as e:
            self.after(0, lambda err=str(e): messagebox.showerror("Ошибка", err))
        finally:
            self.after(0, lambda: self.process_btn.config(state='normal'))

    def process_images_with_progress(self):
        """Основной процесс обработки с обновлением прогресса"""
        try:
            image_files = get_image_files(INPUT_DIR)
            if not image_files:
                raise ValueError(f"В директории {INPUT_DIR} не найдено изображений")

            total_images = len(image_files)
            self.after(0, lambda: self.progress.configure(maximum=total_images))

            processed_images = []
            for i, img_path in enumerate(image_files, 1):
                cropped = process_image(img_path)
                if cropped is not None:
                    processed_images.append(cropped)
                    print(f"Обработано: {img_path.name}")
                self.after(0, self.update_progress, i)

            if processed_images:
                combined = combine_images(processed_images)
                if combined is not None:
                    cv2.imwrite(OUTPUT_PATH, combined)
                    print(f"Результат сохранён в: {OUTPUT_PATH}")

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ошибка обработки", str(e)))

    def show_preview(self):
        """Окно предпросмотра результатов"""
        try:
            preview_window = tk.Toplevel(self)
            preview_window.title("Полученное изображение")

            img = Image.open(OUTPUT_PATH)

            # Масштабирование изображения
            max_size = (800, 600)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(img)

            label = ttk.Label(preview_window, image=img_tk)
            label.image = img_tk  # Сохраняем ссылку
            label.pack(padx=10, pady=10)

            ttk.Button(preview_window, text="Закрыть", command=preview_window.destroy).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Ошибка просмотра", f"Не удалось загрузить изображение:\n{str(e)}")


# Оригинальные функции обработки изображений
def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in Path(directory).iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions]


def reduce_jpeg_artifacts(img):
    filtered = cv2.medianBlur(img, MEDIAN_BLUR_SIZE)
    return cv2.bilateralFilter(filtered,
                               d=BILATERAL_D,
                               sigmaColor=BILATERAL_SIGMA_COLOR,
                               sigmaSpace=BILATERAL_SIGMA_SPACE)


def extract_center_crop(img):
    h, w = img.shape[:2]
    y_start = max(0, (h - CROP_HEIGHT) // 2)
    x_start = max(0, (w - CROP_WIDTH) // 2)
    return img[y_start:y_start + CROP_HEIGHT,
           x_start:x_start + CROP_WIDTH]


def process_image(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Ошибка загрузки: {img_path.name}")
            return None

        processed = reduce_jpeg_artifacts(img)
        cropped = extract_center_crop(processed)
        return cropped
    except Exception as e:
        print(f"Ошибка обработки {img_path.name}: {str(e)}")
        return None


def combine_images(images):
    if not images:
        return None

    min_height = min(img.shape[0] for img in images)
    cropped = [img[:min_height] for img in images]
    return np.hstack(cropped)


if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()