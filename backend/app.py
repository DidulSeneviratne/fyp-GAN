import glob
from http.client import HTTPException
from typing import List
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
from utils import preprocess_image, load_model, generate_ui_image, change_background, resize_to_input, overlay_text_on_image

app = FastAPI()

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Use ["http://localhost:8001"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

OUTPUT_FOLDER = '../backend/outputs'

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
MODEL_PATH = "../backend/models/G_sketch_to_ui.pth"
model = load_model(MODEL_PATH)

font_details_csv = pd.read_csv("font_details.csv")  # Replace with your file path
color_preferences_csv = pd.read_csv("color_preferences.csv")  # Replace with your file path

@app.post("/api/generate-ui")
async def generate_ui(
    sketch: List[UploadFile] = File(...),
    region: str = Form(...),
    age: str = Form(...),
    device: str = Form(...),
    product: str = Form(...)
):
    
    if len(sketch) > 5:
        return JSONResponse(
            content={"error": "You can upload a maximum of 5 images at a time."},
            status_code=400
        )

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

    print(f"{region},{age},{product}")

    if font_data.empty or color_data.empty:
        return JSONResponse(
            content={"error": "No matching data found in the CSV files."},
            status_code=404
        )
    
    font_face = font_data.iloc[0]["Font Face"]
    font_size = font_data.iloc[0]["Font Size"]
    color_preference = tuple(map(int, color_data.iloc[0]["Color Preference"].split(", ")))
    text_color = tuple(map(int, color_data.iloc[0]["Text Color"].split(", ")))

    print(f"{font_face},{font_size},{color_preference},{text_color}")

    for idx, sketches in enumerate(sketch):
        # Read the uploaded image file
        sketch_data = await sketches.read()

        image = Image.open(io.BytesIO(sketch_data))

        # Preprocess the sketch for the model
        preprocessed_image = preprocess_image(sketch_data)

        # Step 2: Generate the UI using the model
        with torch.no_grad():
            generated_tensor = model(preprocessed_image)
        
        # Convert PyTorch tensor to NumPy array if necessary
        preprocessed_image = preprocessed_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW to HWC

        # Generate and save the output image
        output_filename = f"../backend/outputs/generated_ui{idx}.png"
        output_path = generate_ui_image(generated_tensor, output_filename)
        
        # Save the generated image for further processing
        # cv2.imwrite(output_path, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

        # Step 3: Post-process the generated image
        # 3.1 Change Background Color
        generated_image_path = change_background(output_path, color_preference, output_filename)

        # 3.2 Resize the image to match the input dimensions
        resized_image_path = resize_to_input(image, generated_image_path, output_filename)

        # 3.3 Extract and overlay text
        final_image_path = overlay_text_on_image(image, resized_image_path, font_face, font_size, text_color, output_filename)

        generated_images.append(final_image_path)  # Add the final image path to the list
        
    # Return the generated image file
    # return FileResponse(final_image_path, media_type="image/jpeg")
    return JSONResponse(content={"generated_images": generated_images})
