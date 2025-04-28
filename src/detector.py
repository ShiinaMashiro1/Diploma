import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Настройки анализа
PERCENTILE = 99  # Уровень перцентиля (можно менять)
MIN_AREA = 20  # Минимальный размер области


def find_bright_regions(gray_img, percentile):
    """Находит контуры ярких областей выше указанного перцентиля"""
    non_black_mask = gray_img > 0

    if np.any(non_black_mask):
        threshold = np.percentile(gray_img[non_black_mask], percentile)
    else:
        return [], 0, np.zeros_like(gray_img)

    _, binary_mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA or np.max(
                gray_img[cv2.boundingRect(cnt)[1]:cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3],
                cv2.boundingRect(cnt)[0]:cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2]]) == 0:
            continue
        valid_contours.append(cnt)

    return valid_contours, threshold, binary_mask


def process_image(img_path, output_dir, log_file):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        log_file.write(f"ОШИБКА: Не удалось загрузить {img_path.name}\n")
        return

    contours, threshold, mask = find_bright_regions(gray, PERCENTILE)
    marked_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Рисуем точные контуры вместо прямоугольников
    cv2.drawContours(marked_img, contours, -1, (0, 255, 0), 1)

    # Запись информации о контурах
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write(f"Изображение: {img_path.name}\n")
    log_file.write(f"Порог: {threshold:.1f} | Областей: {len(contours)}\n")

    for i, cnt in enumerate(contours, 1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        log_file.write("\n" + "-" * 50 + "\n")
        log_file.write(f"Область {i}:\n")
        log_file.write(f"Площадь: {area} px²\n")
        log_file.write(f"Периметр: {perimeter:.1f} px\n")
        log_file.write(f"Bounding Box: [{x}, {y}, {w}, {h}]\n")
        log_file.write(f"Координаты контура: {len(cnt)} точек\n")

    output_path = output_dir / f"contour_{img_path.name}"
    cv2.imwrite(str(output_path), marked_img)
    log_file.write(f"\nСохранено: {output_path.name}\n")


def detect():
    input_dir = Path("E:\CMU\Code\Greyscale\Grayscale_Output")
    output_dir = Path("E:\CMU\Code\Samples_detected\Bright98_Output")
    output_dir.mkdir(exist_ok=True)

    log_path = output_dir / "Brightness_data.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Отчет: Точные контуры ({PERCENTILE} перцентиль)\n")
        log_file.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for img_path in input_dir.glob('*'):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                process_image(img_path, output_dir, log_file)

        log_file.write("\n" + "=" * 60 + "\n")
        log_file.write("Анализ завершен\n")

    print(f"Результаты в: {output_dir}")

