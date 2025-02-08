import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from models import Generator
import cv2
import easyocr
import numpy as np
import io

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


def overlay_text_on_image(input_image, generated_image_path, face, size, color, final_image_path):
    # Load the generated image
    generated_cv = cv2.imread(generated_image_path)
    generated_rgb = cv2.cvtColor(generated_cv, cv2.COLOR_BGR2RGB)

    # Convert to PIL for text rendering
    pil_image = Image.fromarray(generated_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Extract text from the input image
    extracted_text = easyocr_text_detection(input_image)

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
            print(f"{face}")
            font_path = "/fonts/arial.ttf"  # Replace with your font path
        elif face == 'Verdana':
            print(f"{face}")
            font_path = "/fonts/verdana.ttf"  # Replace with your font path
        elif face == 'Georgia':
            print(f"{face}")
            font_path = "/fonts/georgia.ttf"  # Replace with your font path
        elif face == 'Comic Sans':
            print(f"{face}")
            font_path = "/fonts/comic.ttf"  # Replace with your font path
        elif face == 'Roboto':
            print(f"{face}")
            font_path = "/fonts/verdana.ttf"  # Replace with your font path
        elif face == 'Courier New':
            print(f"{face}")
            font_path = "/fonts/cour.ttf"  # Replace with your font path
        elif face == 'Times New Roman':
            print(f"{face}")
            font_path = "/fonts/times.ttf"  # Replace with your font path

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

