import cv2
import os

def pencil_sketch(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = 255 - gray_image

    # Apply Gaussian blur to the inverted image
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)

    # Invert the blurred image
    inverted_blurred = 255 - blurred

    # Create the pencil sketch by blending the grayscale image with the inverted blurred image
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    return sketch

def convert_images_in_folder(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(input_path)

            # Apply pencil sketch effect
            sketch = pencil_sketch(image)

            # Save the sketch in the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, sketch)
            print(f"Converted {filename} and saved to {output_path}")

# Example usage
input_folder = "UI"
output_folder = "dataset2"
convert_images_in_folder(input_folder, output_folder)