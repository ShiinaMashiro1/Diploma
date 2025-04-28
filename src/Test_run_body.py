import cv2
import numpy as np
from pathlib import Path

from src.Grey_fade import grey_scale
from src.Wavelegth_calc import graphscalc
from src.detector import detect

# ========== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ==========
INPUT_DIR = "E:\CMU\Code\Sky_samples"  # Папка с исходными изображениями
OUTPUT_PATH = "E:\CMU\Code\Samples_res\output_test.jpg"  # Путь для сохранения результата

# Параметры фильтрации артефактов JPEG
MEDIAN_BLUR_SIZE = 19  # Размер ядра медианного фильтра (нечётное число)
BILATERAL_D = 2 # Диаметр соседних пикселей для билатерального фильтра
BILATERAL_SIGMA_COLOR = 10  # Стандартное отклонение в цветовом пространстве
BILATERAL_SIGMA_SPACE = 10  # Стандартное отклонение в координатном пространстве

# Параметры вырезаемой области
CROP_WIDTH = 30  # Ширина вырезаемого фрагмента (X)
CROP_HEIGHT = 2000  # Высота вырезаемого фрагмента (Y)


# ==========================================

def get_image_files(directory):
    """Получение списка изображений в директории"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in Path(directory).iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions]


def reduce_jpeg_artifacts(img):
    """Уменьшение артефактов JPEG"""
    filtered = cv2.medianBlur(img, MEDIAN_BLUR_SIZE)
    return cv2.bilateralFilter(filtered,
                               d=BILATERAL_D,
                               sigmaColor=BILATERAL_SIGMA_COLOR,
                               sigmaSpace=BILATERAL_SIGMA_SPACE)


def extract_center_crop(img):
    """Вырезание центрального фрагмента"""
    h, w = img.shape[:2]
    y_start = max(0, (h - CROP_HEIGHT) // 2)
    x_start = max(0, (w - CROP_WIDTH) // 2)

    return img[y_start:y_start + CROP_HEIGHT,
           x_start:x_start + CROP_WIDTH]


def process_image(img_path):
    """Обработка одного изображения"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Ошибка загрузки: {img_path.name}")
            return None

        # Предобработка изображения
        processed = reduce_jpeg_artifacts(img)

        # Вырезание центрального фрагмента
        cropped = extract_center_crop(processed)

        return cropped

    except Exception as e:
        print(f"Ошибка обработки {img_path.name}: {str(e)}")
        return None


def combine_images(images):
    """Склейка изображений в одну полосу"""
    if not images:
        return None

    # Определение минимальной высоты
    min_height = min(img.shape[0] for img in images)

    # Обрезка изображений до одинаковой высоты
    cropped = [img[:min_height] for img in images]

    return np.hstack(cropped)


def main():
    # Получение списка изображений
    image_files = get_image_files(INPUT_DIR)
    if not image_files:
        print(f"В директории {INPUT_DIR} не найдено изображений")
        return

    print(f"Найдено изображений: {len(image_files)}")

    # Обработка всех изображений
    processed_images = []
    for img_path in image_files:
        cropped = process_image(img_path)
        if cropped is not None:
            processed_images.append(cropped)
            print(f"Обработано: {img_path.name}")

    # Склейка и сохранение результата
    if processed_images:
        combined = combine_images(processed_images)
        if combined is not None:
            cv2.imwrite(OUTPUT_PATH, combined)
            print(f"\nРезультат сохранён в: {OUTPUT_PATH}")
            print(f"Финальный размер: {combined.shape[1]}x{combined.shape[0]}")
    else:
        print("Не удалось обработать ни одного изображения")


if __name__ == "__main__":
    main()
    grey_scale()
    detect()
    graphscalc()