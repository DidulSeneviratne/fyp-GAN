import os
import io
import cv2
import faiss
import torch
import easyocr
import numpy as np
import torchvision.transforms as transforms
from models import Generator
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
from torchvision.models import resnet50, ResNet50_Weights
from skimage.metrics import structural_similarity as ssim


def resize_image(image_path, target_size):
    with Image.open(image_path) as img:
        resized_img = img.resize(target_size)
        resized_img.save(image_path)

    return image_path
        

def resize_imagePL(image_path, orientation, device):
    widthM, heightM = 1920, 1080
    widthT, heightT = 1200, 1920

    with Image.open(image_path) as img:
        if orientation == "Portrait" and device == "Tablet":
            resized_img = img.resize((widthT, heightT))
            resized_img.save(image_path)  # overwrite original
        elif orientation == "Landscape" and device == "Mobile":
            resized_img = img.resize((widthM, heightM))
            resized_img.save(image_path)  # overwrite original
        # else do nothing

    return image_path
    

def rgb_to_hsv(rgb):
    # Convert RGB to a numpy array and reshape it to match OpenCV's input format
    rgb_color = np.uint8([[list(rgb)]])  # Convert (R, G, B) to 3D numpy array

    # Convert from RGB to HSV
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)

    # Extract HSV values
    return tuple(hsv_color[0][0])  # Return as a tuple (H, S, V)


def load_model(model_path):
    """Load the Generator model with pre-trained weights."""
    # Initialize the Generator model
    model = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # Set the model to evaluation mode
    model.eval()
    return model


def preprocess_image(image_bytes):
    """Preprocess the uploaded sketch image."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)


def generate_ui_image(output_tensor, output_path):
    """Convert the model output tensor to an image and save it."""
    tensor = (output_tensor.squeeze().cpu() + 1) / 2
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)
    return output_path


def change_background(image_path, color_preference, output_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define white range for background detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create a mask for white areas
    mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Change the background to yellow
    hue = color_preference[0]
    if hue > 255:
        hue = hue % 256
    hsv_image[mask > 0, 0] = hue
    hsv_image[mask > 0, 1] = color_preference[1]
    hsv_image[mask > 0, 2] = color_preference[2]

    # Convert back to BGR
    processed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Save and return the path
    cv2.imwrite(output_path, processed_image)
    return output_path


def resize_to_input(input_image, generated_image_path, output_path):
    # Load images
    input_cv = np.array(input_image)
    generated_cv = cv2.imread(generated_image_path)

    # Get dimensions of the input image
    input_height, input_width = input_cv.shape[:2]

    # Resize the generated image to match input dimensions
    resized_image = cv2.resize(generated_cv, (input_width, input_height), interpolation=cv2.INTER_AREA)

    # Save and return the path
    # output_path = "outputs/resized_image.jpg"
    cv2.imwrite(output_path, resized_image)
    return output_path


def easyocr_text_detection(image_path):
    # Initialize the EasyOCR Reader
    reader = easyocr.Reader(['en'])  # Add other languages as needed, e.g., ['en', 'es']

    # Perform OCR on the image
    results = reader.readtext(image_path)

    # Extract detected text
    extracted_text = []
    for result in results:
        bbox, text, confidence = result
        extracted_text.append((text, confidence, bbox))

    return extracted_text


def overlay_text_on_image(input_image, generated_image_path, face, size, color, final_image_path, sketch_data):
    # Load the generated image
    generated_cv = cv2.imread(generated_image_path)
    generated_rgb = cv2.cvtColor(generated_cv, cv2.COLOR_BGR2RGB)

    # Convert to PIL for text rendering
    pil_image = Image.fromarray(generated_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Extract text from the input image
    extracted_text = easyocr_text_detection(sketch_data)

    # Overlay text
    for text, _, bbox in extracted_text:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        x3, y3 = bbox[2] 
        x4, y4 = bbox[3]

        # Dynamic font size based on bounding box height
        box_width = int(x2 - x1)
        box_height = int(y3 - y1)
        font_size = max(size, int(box_height * 0.7))
        #font_path = "/fonts/roboto.ttf"
        if face == 'Arial':
            # print(f"{face}")
            font_path = "/fonts/arial.ttf"  # Replace with your font path
        elif face == 'Verdana':
            # print(f"{face}")
            font_path = "/fonts/verdana.ttf"  # Replace with your font path
        elif face == 'Georgia':
            # print(f"{face}")
            font_path = "/fonts/georgia.ttf"  # Replace with your font path
        elif face == 'Comic Sans':
            # print(f"{face}")
            font_path = "/fonts/comic.ttf"  # Replace with your font path
        elif face == 'Roboto':
            # print(f"{face}")
            font_path = "/fonts/arial.ttf"  # Replace with your font path
        elif face == 'Courier New':
            # print(f"{face}")
            font_path = "/fonts/cour.ttf"  # Replace with your font path
        elif face == 'Times New Roman':
            # print(f"{face}")
            font_path = "/fonts/times.ttf"  # Replace with your font path
        elif face == 'Serif':
            # print(f"{face}")
            font_path = "/fonts/times.ttf"  # Replace with your font path
        elif face == 'Sans-serif':
            # print(f"{face}")
            font_path = "/fonts/verdana.ttf"  # Replace with your font path
        elif face == 'Garamond':
            # print(f"{face}")
            font_path = "/fonts/georgia.ttf"  # Replace with your font path 
        elif face == 'Helvetica':
            # print(f"{face}")
            font_path = "/fonts/arial.ttf"  # Replace with your font path

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text dimensions
        text_width, text_height = font.getmask(text).size    

        # Define text color, shadow, and background box properties
        text_color = color  # Yellow text
        if text_color == (255, 255, 255): 
            shadow_color = (0, 0, 0)  # Black shadow
            background_color = (0, 0, 0, 128)  # Semi-transparent black box (RGBA for PIL)
        else:
            shadow_color = (255, 255, 255)  # Black shadow
            background_color = (255, 255, 255, 128)  # Semi-transparent black box (RGBA for PIL)

        # Step 1: Draw a semi-transparent box behind the text
        box_x1 = x1 - 5  # Add padding
        box_y1 = y1 - 5
        box_x2 = x1 + text_width + 5
        box_y2 = y1 + text_height + 5
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=background_color)

        # Step 2: Add a shadow for the text
        shadow_offset = 2  # Adjust as needed
        draw.text((x1 + shadow_offset, y1 + shadow_offset), text, fill=shadow_color, font=font)

        # Step 3: Draw the actual text on top
        draw.text((x1, y1), text, fill=text_color, font=font)

    # Convert back to OpenCV format and save
    final_image = np.array(pil_image)
    # final_image_path = "outputs/final_image.jpg"
    cv2.imwrite(final_image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    return final_image_path

def extract_features(image_path):
    """Extract deep learning features from an image using ResNet-50."""

    # Load a pre-trained ResNet-50 model for feature extraction
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()

    # Image preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

def calculate_ssim_parallel(args):
    """Helper function for multiprocessing SSIM calculation."""
    img1_path, img2_path = args
    return calculate_ssim(img1_path, img2_path)


def compute_ssim_parallel(low_quality_img_path, dataset_image_paths):
    """Compute SSIM in parallel across multiple CPU cores."""
    args = [(low_quality_img_path, img_path) for img_path in dataset_image_paths]
    
    with Pool(processes=cpu_count()) as pool:
        ssim_scores = pool.map(calculate_ssim_parallel, args)

    return ssim_scores


def build_faiss_index(feature_list):
    """Builds a FAISS index for fast nearest neighbor search."""
    d = feature_list.shape[1]  # Feature vector dimension
    index = faiss.IndexFlatL2(d)  # L2 (Euclidean distance) index
    index.add(feature_list.astype('float32'))  # Add dataset features to FAISS index
    return index


def search_faiss(index, query_vector, image_paths, k=1):
    """Searches for the k most similar images in the FAISS index."""
    query_vector = np.expand_dims(query_vector, axis=0).astype('float32')  # Ensure correct shape
    D, I = index.search(query_vector, k)  # Find nearest neighbors
    return [image_paths[i] for i in I[0]]  # Return the most similar image paths


def reduce_dimensions(features, n_components=128):
    """Reduces feature dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


def find_most_similar_image_optimized(low_quality_img_path, dataset_folder):
    """Optimized function to find the most similar image using FAISS and parallel SSIM."""
    
    # Load dataset features (or extract if not saved)
    try:
        dataset_features = np.load("image_features.npy")
        image_paths = np.load("image_paths.npy", allow_pickle=True)
    except:
        dataset_features = []
        image_paths = []
        for img_name in os.listdir(dataset_folder):
            img_path = os.path.join(dataset_folder, img_name)
            img_features = extract_features(img_path)
            dataset_features.append(img_features)
            image_paths.append(img_path)

        dataset_features = np.array(dataset_features)
        np.save("image_features.npy", dataset_features)  # Cache feature vectors
        np.save("image_paths.npy", np.array(image_paths))  # Cache image paths

    # Reduce dimensions using PCA
    pca = PCA(n_components=50) # Create PCA object outside the loop
    dataset_features = pca.fit_transform(dataset_features) # Fit and transform dataset feature

    # Extract features for the input image and reduce dimensions
    low_quality_features = extract_features(low_quality_img_path)
    low_quality_features = pca.transform(np.array([low_quality_features]))[0]  # Transform using fitted PCA

    # Use FAISS for fast retrieval
    faiss_index = build_faiss_index(dataset_features)
    top_matches = search_faiss(faiss_index, low_quality_features, image_paths, k=5)

    # Compute SSIM in parallel for the top 5 similar images
    ssim_scores = compute_ssim_parallel(low_quality_img_path, top_matches)

    # Select the best match based on SSIM
    best_ssim_idx = np.argmax(ssim_scores)
    best_match_path = top_matches[best_ssim_idx]

    print(f"Best Match (FAISS + SSIM): {best_match_path}")
    print(f"SSIM Score: {ssim_scores[best_ssim_idx]}")

    return best_match_path


# Identify the background color
def find_dominant_color(image, k=4):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k).fit(pixels)
    dominant_color = kmeans.cluster_centers_[kmeans.labels_[0]].astype(int)
    return tuple(dominant_color)


def change_background_color(image, new_color):
    # Convert the background_color to a 3-channel RGB tuple
    image_path = cv2.imread(image)
    rgb_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    background_color = find_dominant_color(rgb_image)
    background_color_rgb = background_color[:3]

    # Calculate lower and upper boundaries for inRange
    lower_boundary = np.array(background_color_rgb) - 30
    upper_boundary = np.array(background_color_rgb) + 30

    mask = cv2.inRange(image, lower_boundary, upper_boundary)  # Tolerance range
    image[mask > 0] = new_color  # Replace the background
    return image


def enhance_contrast(image):
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Merge the LAB channels back
    enhanced_image = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    return enhanced_image


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
    best_match = low_quality_img_path
    best_euclidean_score = float('inf')  # Lower is better
    best_ssim_score = -1  # Higher is better
    best_combined_score = float('inf')

    # Iterate over dataset images
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)

        # Extract features for dataset image
        img_features = extract_features(img_path)

        # Calculate Euclidean Distance
        # euclidean_score = euclidean(low_quality_features, img_features)
        euclidean_score = np.linalg.norm(low_quality_features - img_features)

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

    # print(f"Most similar image: {best_match}")
    print(f"Euclidean Distance: {best_euclidean_score}")
    print(f"SSIM Score: {best_ssim_score}")

    return best_match