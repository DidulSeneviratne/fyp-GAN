import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
import cv2

# Load a pre-trained ResNet-50 model for feature extraction
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()

# Image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Extract deep learning features from an image using ResNet-50."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

def calculate_ssim(image_path1, image_path2):
    """Calculate SSIM (Structural Similarity Index) between two images."""
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same size if they are different
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    # Compute SSIM
    score, _ = ssim(img1, img2, full=True)
    return score

def find_most_similar_image(low_quality_img_path, dataset_path):
    """Find the most similar high-quality image using Euclidean Distance & SSIM."""
    
    # Extract features for the low-quality image
    low_quality_features = extract_features(low_quality_img_path)

    # Store results
    best_match = None
    best_euclidean_score = float('inf')  # Lower is better
    best_ssim_score = -1  # Higher is better
    best_combined_score = float('inf')

    # Iterate over dataset images
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)

        # Extract features for dataset image
        img_features = extract_features(img_path)

        # Calculate Euclidean Distance
        euclidean_score = euclidean(low_quality_features, img_features)

        # Calculate SSIM Score
        ssim_score = calculate_ssim(low_quality_img_path, img_path)

        # Normalize SSIM Score (higher is better, so we subtract from 1)
        combined_score = euclidean_score - (ssim_score * 100)  # Weighted scoring

        # Find the best match based on combined metric
        if combined_score < best_combined_score:
            best_combined_score = combined_score
            best_match = img_path
            best_euclidean_score = euclidean_score
            best_ssim_score = ssim_score

    print(f"Most similar image: {best_match}")
    print(f"Euclidean Distance: {best_euclidean_score}")
    print(f"SSIM Score: {best_ssim_score}")

    return best_match

# Example Usage
dataset_folder = "dataset3"
low_quality_image = "dataset1/46.jpg"

best_match_image = find_most_similar_image(low_quality_image, dataset_folder)
