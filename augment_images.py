import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def augment_images(input_folder, output_folder, num_augmentations_per_image=5):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

    # Loop through all images in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate augmented images
            for i in range(num_augmentations_per_image):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                # Save the augmented image
                output_filename = f"aug_{i}_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# Usage
input_folder = r"FILEPATH"
output_folder = r"FILEPATH"
num_augmentations_per_image = 3

print(f"Augmenting images from: {input_folder}")
print(f"Saving augmented images to: {output_folder}")

augment_images(input_folder, output_folder, num_augmentations_per_image)

print("Augmentation complete!")
