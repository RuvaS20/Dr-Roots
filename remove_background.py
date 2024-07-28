import cv2
import numpy as np
from skimage import filters
import os
from PIL import Image

def remove_background(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's method for thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find the largest contour (assuming it's the leaf)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask from the largest contour
        mask = np.zeros(cleaned.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], 0, (255), -1)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        
        # Convert black background to transparent
        rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        
        # Save the result
        Image.fromarray(rgba).save(output_path)
    else:
        print(f"No contours found in {image_path}")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename.split('.')[0]}.png")
            remove_background(input_path, output_path)
            print(f"Processed {filename}")

# Usage 
base_input_dir = r"DIRECTORY TO TAKE IMAGES FROM"
base_output_dir = r"DIRECTORY TO SEND IMAGES TO"

print(f"Processing images from: {base_input_dir}")
print(f"Saving processed images to: {base_output_dir}")

process_directory(base_input_dir, base_output_dir)
print("Processing complete!")
