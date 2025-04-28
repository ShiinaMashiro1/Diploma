import cv2
import numpy as np
from pathlib import Path

from src.Grey_fade import grey_scale
from src.detector import detect

stripw = 3


def get_image_files(directory):
    """Get all image files in the specified directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in Path(directory).iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions]


def extract_central_strip(img):
    """Extract 3-pixel-wide vertical strip from image center"""
    center_x = img.shape[1] // 2
    x = max(0, center_x - 1)
    return img[:, x:x + stripw].copy()  # Get 3 columns from center


def combine_strips(strips):
    """Combine all strips horizontally"""
    min_height = min(s.shape[0] for s in strips)
    total_width = sum(s.shape[1] for s in strips)

    # Create blank canvas (handles both color and grayscale)
    if strips[0].ndim == 3:
        combined = np.zeros((min_height, total_width, 3), dtype=np.uint8)
    else:
        combined = np.zeros((min_height, total_width), dtype=np.uint8)

    x = 0
    for strip in strips:
        h = min(min_height, strip.shape[0])
        combined[:h, x:x + strip.shape[1]] = strip[:h]
        x += strip.shape[1]
    return combined

def apply_jpg_artifact_reduction(img):
    """Фильтрация артефактов сжатия JPEG"""
    # Медианный фильтр для уменьшения блочных артефактов
    filtered = cv2.medianBlur(img, 1)

    # Билатеральная фильтрация для сохранения границ
    return cv2.bilateralFilter(filtered, d=1, sigmaColor=10, sigmaSpace=10)


def validate_strip(strip):
    """Проверка и коррекция аномальных пикселей в полосе"""
    # Замена выбросов, отличающихся больше чем на 30 от медианы
    median = np.median(strip, axis=(0, 1))
    mask = np.any(np.abs(strip.astype(np.int16) - median > 30, axis=2))
    strip[mask] = median
    return strip


def main():
    input_dir = "E:\CMU\Code\Sky_samples"
    output_path = "E:\CMU\Code\Samples_res\output.jpg"

    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'")

    strips = []
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Конвертация в BGR если нужно
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        try:
            strip = extract_central_strip(img)
            strips.append(strip)
            print(f"Processed: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

    if not strips:
        print("No valid strips processed")
        return

    # Сохранение с максимальным качеством
    result = combine_strips(strips)
    cv2.imwrite(output_path, result,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100,  # Максимальное качество
                 int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
    print(f"Saved combined image to '{output_path}'")


if __name__ == "__main__":
    main()
    grey_scale()
    detect()