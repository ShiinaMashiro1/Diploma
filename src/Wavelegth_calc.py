import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Глобальные переменные для настройки
STRIPE_WIDTH = 30  # Ширина полоски в пикселях
INPUT_PATH = 'E:\CMU\Code\Samples_res\output_test.jpg'  # Путь к изображениям
OUTPUT_DIR = 'E:\CMU\Code\Graphics'  # Папка для сохранения графиков


def rgb_to_wavelength(rgb):
    """
    Улучшенное преобразование RGB в длину волны.
    Учитывает соотношение каналов и физические пределы.
    """
    r, g, b = rgb.astype(float) / 255.0
    eps = 1e-6  # Для избежания деления на ноль

    # Нормализация и определение доминирующего канала
    total = r + g + b + eps
    r_norm, g_norm, b_norm = r / total, g / total, b / total
    dominant = np.argmax([r_norm, g_norm, b_norm])

    # Расчет длины волны с учетом цветового вклада
    if dominant == 0:  # Красный доминирует
        ratio = (r - max(g, b)) / (r + eps)
        wavelength = 630 + 70 * ratio  # 630-700 нм
    elif dominant == 1:  # Зеленый доминирует
        ratio = (g - max(r, b)) / (g + eps)
        wavelength = 520 + 50 * ratio  # 520-570 нм
    else:  # Синий доминирует
        ratio = (b - max(r, g)) / (b + eps)
        wavelength = 450 + 40 * ratio  # 450-490 нм

    # Ограничение физических пределов
    return np.clip(wavelength, 380, 750)


def process_keogram(image_path, output_dir):
    """Обработка кеограммы с поиском пиковых длин волн."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    wavelengths = []

    for x_start in range(0, width, STRIPE_WIDTH):
        x_end = min(x_start + STRIPE_WIDTH, width)
        stripe_gray = gray[:, x_start:x_end]

        # Поиск координаты пикового пикселя
        y_max, x_local = np.unravel_index(
            np.argmax(stripe_gray), stripe_gray.shape
        )
        x_global = x_start + x_local

        # Получение RGB пикового пикселя
        peak_rgb = img_rgb[y_max, x_global]
        wavelength = rgb_to_wavelength(peak_rgb)
        wavelengths.append(wavelength)

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(wavelengths)), wavelengths, 'b-', marker='o', markersize=3)
    plt.title('Пиковые длины волн по времени')
    plt.xlabel('Номер полоски (время)')
    plt.ylabel('Длина волны (нм)')
    plt.ylim(380, 750)
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '_plot.png'))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def graphscalc():
    """Обработка всех изображений в папке."""
    image_paths = glob.glob(INPUT_PATH)
    if not image_paths:
        print(f"Изображения не найдены: {INPUT_PATH}")
        return

    for path in image_paths:
        process_keogram(path, OUTPUT_DIR)
    print(f"Графики сохранены в: {os.path.abspath(OUTPUT_DIR)}")


    graphscalc()