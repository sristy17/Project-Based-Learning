import cv2
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


    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_value = ssim_map.mean()

    return ssim_value


image1 = cv2.imread("C:/PBL_FINAL/Project-Based-Learning/Pancreas/10x data/1_Pancreas_10x.tif") 
image2 = cv2.imread("C:/PBL_FINAL/Project-Based-Learning/Pancreas/10x data/2_Pancreas_10x.tif") 


new_width = 640
new_height = 480

scaled_image1 = cv2.resize(image1, (new_width, new_height), interpolation=cv2.INTER_AREA)

ssim_value = calculate_ssim(scaled_image1, image2)
print(f"SSIM: {ssim_value}")
