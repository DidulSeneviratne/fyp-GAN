import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

# Identify the background color
def find_dominant_color(image, k=4):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    dominant_color = kmeans.cluster_centers_[kmeans.labels_[0]].astype(int)
    return tuple(dominant_color)

def change_background_color(image, new_color):

    # Ensure image_path is a valid string
    if not isinstance(image, str) or not image.strip():
        raise ValueError("Error: `image_path` must be a valid file path (string).")

    # Ensure file exists before reading
    if not os.path.exists(image):
        raise FileNotFoundError(f"Error: Image file not found at {image}")

     # Read the image
    image = cv2.imread(image)
    if image is None:
        raise ValueError(f"Error: Could not read image from {image}. Check the file path.")

    # Convert the background_color to a 3-channel RGB tuple
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_path = cv2.imread(image)
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    background_color = find_dominant_color(rgb_image)
    background_color_rgb = background_color[:3]

    # Calculate lower and upper boundaries for inRange
    lower_boundary = np.array(background_color_rgb) - 30
    upper_boundary = np.array(background_color_rgb) + 30

    mask = cv2.inRange(image, lower_boundary, upper_boundary)  # Tolerance range
    image[mask > 0] = new_color  # Replace the background

    # Generate a new filename
    # base_dir, filename = os.path.split(image)  # Get directory and filename
    #name, ext = os.path.splitext(filename)  # Extract name and extension
    new_filename = f"1_modified.jpg"  # Example: 1_modified.jpg
    output_path = os.path.join("dataset1/", new_filename)

    # Save the new image
    cv2.imwrite(output_path, image)

    return image

text_color = (255,255,0)

output_path = 'dataset1/1.jpg'

image0 = change_background_color(output_path, text_color)