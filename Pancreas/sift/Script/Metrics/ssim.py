import cv2
import os
import numpy as np

def calculate_ssim(img1, img2):
    
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    
    C1 = 6.5025
    C2 = 58.5225

    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_value = ssim_map.mean()

    return ssim_value


folder1 = 'C:/PBL_FINAL/Project-Based-Learning/Pancreas/10x_png'           # Folder with registered images (image1)
folder2 = 'C:/PBL_FINAL/Project-Based-Learning/Pancreas/Overlaid_SIFT' # Folder with input images (image2)


new_width = 640
new_height = 480




for i in range(2, 36):
    image1_base_name = f"{i}_Pancreas_10x"  # Image from folder1 (with extension)
    image2_base_name = f"{i-1}"          # Corresponding image from folder2 (with extension)

    
    image1_path = f"{folder1}/{image1_base_name}.png"
    image2_path = f"{folder2}/{image2_base_name}.png"

    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print(f"Error loading images: {image1_base_name}, {image2_base_name}")
    else:
    
        scaled_image1 = cv2.resize(image1, (new_width, new_height), interpolation=cv2.INTER_AREA)
        scaled_image2 = cv2.resize(image2, (new_width, new_height), interpolation=cv2.INTER_AREA)

        
        ssim_value = calculate_ssim(scaled_image1, scaled_image2)

        
        print(f"SSIM between {image1_base_name}.png and {image2_base_name}.png: {ssim_value}")
    

