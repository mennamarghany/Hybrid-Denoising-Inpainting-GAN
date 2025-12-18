import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from deepfillv2.utils import psnr  # Assuming this exists in your cloned repo

def denoise_image(image_path):
    """
    Stage 1: Apply Non-Local Means Denoising (Simulating N2V inference for demo).
    In a full prod environment, this would call the N2V model inference.
    """
    print(f"Applying Denoising Stage to {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    
    # Denoise each channel
    b, g, r = cv2.split(img)
    h_val = 10
    b_clean = cv2.fastNlMeansDenoising(b, None, h_val, 7, 21)
    g_clean = cv2.fastNlMeansDenoising(g, None, h_val, 7, 21)
    r_clean = cv2.fastNlMeansDenoising(r, None, h_val, 7, 21)
    
    clean_img = cv2.merge([b_clean, g_clean, r_clean])
    return clean_img

def sharpen_image(img):
    """Optional post-processing refinement."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def run_pipeline(input_path, mask_path, output_path):
    # 1. Denoise
    clean_img = denoise_image(input_path)
    denoised_path = "temp_denoised.jpg"
    cv2.imwrite(denoised_path, clean_img)
    
    # 2. Inpaint (Calling the DeepFill model)
    # Note: In a real script, you would import the model class here.
    # For this portfolio demo, we simulate the shell command trigger.
    print("Running DeepFill v2 Inpainting...")
    exit_code = os.system(f"python inpaint.py --image {denoised_path} --mask {mask_path} --output {output_path}")
    
    if exit_code == 0:
        print(f"✅ Restoration Complete. Saved to {output_path}")
    else:
        print("❌ Inpainting Failed.")

if __name__ == "__main__":
    # Create dummy directories if they don't exist
    os.makedirs("output", exist_ok=True)
    
    # Example Usage
    run_pipeline("input/img.jpg", "input/mask.png", "output/result.jpg")
