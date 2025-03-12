import cv2
import os
import numpy as np

def pencil_sketch(image_path, output_path):
    """Convert a single image to a pencil sketch and save it."""
    img = cv2.imread(image_path)  # Load image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use Laplacian Edge Detection for Sketch Effect
    edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=3)

    # Invert Colors for Sketch Effect
    inverted = 255 - edges

    # Normalize to Enhance Sketch Look
    normalized = cv2.normalize(inverted, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(output_path, normalized)  # Save the sketch

def batch_convert_to_sketch(input_folder, output_folder):
    """Convert all images in a folder to pencil sketches."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            pencil_sketch(input_path, output_path)
            print(f"Processed: {filename}")

# Example Usage
input_folder = "UI"  # Folder containing original images
output_folder = "dataset1"  # Folder to save sketches

batch_convert_to_sketch(input_folder, output_folder)
print("Sketch conversion completed!")