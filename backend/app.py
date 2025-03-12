import glob
from http.client import HTTPException
import json
import shutil
from typing import List
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from utils import preprocess_image, load_model, generate_ui_image, change_background, resize_to_input, overlay_text_on_image, find_most_similar_image, rgb_to_hsv, resize_image

app = FastAPI()

'''app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")'''
# app.mount("/", StaticFiles(directory="outputs", html=True), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],  # Use ["http://localhost:8002"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the outputs directory exists
os.makedirs("output", exist_ok=True)

OUTPUT_FOLDER = '../frontend/output'

@app.delete("/delete-all-images")
async def delete_all_images():
    try:
        # Get all image files inside the outputs folder (PNG, JPG, JPEG)
        image_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*"))

        if not image_files:
            return {"message": "No images found in the outputs folder."}

        # Delete each image file
        for file_path in image_files:
            os.remove(file_path)

        return {"message": "All images have been deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting images: {str(e)}")

# Load the trained model
MODEL_PATH = "models/G_sketch_to_ui.pth"
model = load_model(MODEL_PATH)

font_details_csv = pd.read_csv("font_details.csv")  # Replace with your file path
color_preferences_csv = pd.read_csv("color_preferences.csv")  # Replace with your file path

@app.post("/api/generate-ui")
async def generate_ui(
    sketch: List[UploadFile] = File(...),
    region: str = Form(...),
    age: str = Form(...),
    device: str = Form(...),
    product: str = Form(...),
    useCustomColor: str = Form(...),
    colors1: str = Form(...),
):
    
    if len(sketch) > 5:
        return JSONResponse(
            content={"error": "You can upload a maximum of 5 images at a time."},
            status_code=400
        )
    
    MEAN_RESOLUTIONS = {
        "Desktop": (3072, 1824),
        "Mobile": (880, 1840),
        "Tablet": (1664, 1312)
    }

    SIZE_RANGES = {
        "Desktop": {"min_w": 1024, "max_w": 5120, "min_h": 768, "max_h": 2880},
        "Mobile": {"min_w": 320, "max_w": 1440, "min_h": 480, "max_h": 3200},
        "Tablet": {"min_w": 768, "max_w": 2560, "min_h": 1024, "max_h": 1600},
    }

    generated_images = []
    
    # Extract font details
    font_data = font_details_csv[
        (font_details_csv["Region"] == region) &
        (font_details_csv["Age"] == age) &
        (font_details_csv["Product"] == product)
    ]

    # Extract color preferences
    color_data = color_preferences_csv[
        (color_preferences_csv["Region"] == region) &
        (color_preferences_csv["Age"] == age) &
        (color_preferences_csv["Product"] == product)
    ]

    print(f"{region},{age},{product},{device},{useCustomColor},{colors1}")

    if font_data.empty or color_data.empty:
        return JSONResponse(
            content={"error": "No matching data found in the CSV files."},
            status_code=404
        )
    
    ''' processed_images = []  # Stores resized/valid images as byte arrays

    for file in sketch:
        image_p = Image.open(file.file)
        width, height = image_p.size

        # Check if image is within allowed range
        valid_range = SIZE_RANGES[device]
        if not (valid_range["min_w"] <= width <= valid_range["max_w"] and valid_range["min_h"] <= height <= valid_range["max_h"]):
            image_p = resize_image(image_p, MEAN_RESOLUTIONS[device])  # Resize if out of range

        # Convert resized image back to bytes
        img_io = io.BytesIO()
        img_io.seek(0)

        # Store the image byte array
        processed_images.append(img_io.getvalue()) '''

    cleaned_list1 = json.loads(colors1)  # Converts to a Python list

    # Convert list to a comma-separated string
    result_color1 = ", ".join(cleaned_list1)

    font_face = font_data.iloc[0]["Font Face"]
    font_size = font_data.iloc[0]["Font Size"]

    if(useCustomColor == 'true'):
        colorPreference = tuple(map(int, result_color1.split(", ")))
        color_preference = rgb_to_hsv(colorPreference)
    else:
        color_preference = tuple(map(int, color_data.iloc[0]["Color Preference"].split(", ")))
        
    text_color = tuple(map(int, color_data.iloc[0]["Text Color"].split(", ")))

    print(f"{font_face},{font_size},{color_preference},{text_color}")

    dataset_folder = "dataset3"

    for idx, sketches in enumerate(sketch):

        # Generate a unique temporary filename
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join("temp_images", temp_filename)  # Ensure "temp_images" exists
        os.makedirs("temp_images", exist_ok=True)

        # Save the uploaded image to a local file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(sketches.file, buffer)

        # Now pass the **temporary file path** to the function
        best_match_image = find_most_similar_image(temp_path, dataset_folder)

        print(best_match_image)

        # Open the best match image correctly
        try:
            image = Image.open(best_match_image).convert("RGB")
        except UnidentifiedImageError: # type: ignore
            return {"error": "Invalid matched image file"}

        # Read the uploaded image file
        with open(best_match_image, "rb") as f:
            sketch_data = f.read()

        width, height = image.size

        # Check if image is within allowed range
        valid_range = SIZE_RANGES[device]
        if not (valid_range["min_w"] <= width <= valid_range["max_w"] and valid_range["min_h"] <= height <= valid_range["max_h"]):
            image = resize_image(image, MEAN_RESOLUTIONS[device])  # Resize if out of range

            # Convert resized image back to bytes
            img_io = io.BytesIO()
            image.save(img_io, format="PNG")  # Save as PNG or JPEG as needed
            img_io.seek(0)

            sketch_data = img_io.getvalue()  # Store resized image bytes in sketch_data

        # image = Image.open(io.BytesIO(sketch_data))

        # Preprocess the sketch for the model
        preprocessed_image = preprocess_image(sketch_data)

        # Step 2: Generate the UI using the model
        with torch.no_grad():
            generated_tensor = model(preprocessed_image)
        
        # Convert PyTorch tensor to NumPy array if necessary
        preprocessed_image = preprocessed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW to HWC

        # Generate and save the output image
        output_filename = f"../frontend/output/generated_ui{idx}.png"
        output_path = generate_ui_image(generated_tensor, output_filename)
        
        # Save the generated image for further processing
        # cv2.imwrite(output_path, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

        # Step 3: Post-process the generated image
        # 3.1 Change Background Color
        # image0 = change_background_color(output_path, text_color)
        generated_image_path = change_background(output_path, color_preference, output_filename)

        # 3.2 Resize the image to match the input dimensions
        resized_image_path = resize_to_input(image, generated_image_path, output_filename)

        # 3.3 Extract and overlay text
        final_image_path = overlay_text_on_image(image, resized_image_path, font_face, font_size, text_color, output_filename, sketch_data)

        generated_images.append(final_image_path)  # Add the final image path to the list

        # Cleanup
        os.remove(temp_path)
        
    # Return the generated image file
    # return FileResponse(final_image_path, media_type="image/jpeg")
    return JSONResponse(content={"generated_images": generated_images})
