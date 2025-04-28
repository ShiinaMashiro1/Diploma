import cv2
import numpy as np
from pathlib import Path


def convert_to_grayscale(color_img):
    """
    Convert color image to grayscale while preserving perceived brightness.
    Uses the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    """
    if len(color_img.shape) == 2:  # Already grayscale
        return color_img.copy()

    # Convert BGR to RGB (OpenCV uses BGR by default)
    rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    # Calculate luminance using the standard formula
    grayscale = np.dot(rgb_img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    return grayscale


def process_images(input_dir, output_dir):
    """Process all images in the input directory"""
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the input directory
    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in valid_extensions:
            try:
                # Read the image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read image {img_path.name}")
                    continue

                # Convert to grayscale
                gray_img = convert_to_grayscale(img)

                # Save the result
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), gray_img)
                print(f"Converted: {img_path.name}")

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")


def grey_scale():
    # Set paths
    input_dir = Path("E:\CMU\Code\Samples_res")  # Folder containing color images
    output_dir = Path("E:\CMU\Code\Greyscale\Grayscale_Output")  # Folder for grayscale results

    print(f"Converting images from: {input_dir}")
    print(f"Saving grayscale images to: {output_dir}")

    # Process all images
    process_images(input_dir, output_dir)

    print("Conversion complete!")


