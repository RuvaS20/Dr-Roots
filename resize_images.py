import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(224, 224)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                img_resized = img.resize(size, Image.LANCZOS)
                
                # Prepare the output filename
                output_filename = f"resized_{filename.split('.')[0]}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # Save the resized image
                img_resized.save(output_path)
                print(f"Resized {filename} to 224x224")

# Usage
input_folder = r"FILEPATH"
output_folder = r"FILEPATH"

print(f"Resizing images from: {input_folder}")
print(f"Saving resized images to: {output_folder}")

resize_images(input_folder, output_folder)

print("Resizing complete!")
