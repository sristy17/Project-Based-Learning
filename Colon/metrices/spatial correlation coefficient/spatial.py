import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_and_prepare_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same dimensions")
    
    return image1, image2

def downsample_image(image, factor):
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_AREA)

def generate_random_points(image, num_points, seed=42):
    np.random.seed(seed)  # Set the random seed for reproducibility
    height, width = image.shape
    y_coords = np.random.randint(0, height, size=num_points)
    x_coords = np.random.randint(0, width, size=num_points)
    return list(zip(x_coords, y_coords))

def calculate_good_matches(points1, points2, image1, image2, good_match_threshold=30):
    good_matches = []

    for (x1, y1), (x2, y2) in zip(points1, points2):
        # Get pixel values
        pixel_value1 = image1[y1, x1]
        pixel_value2 = image2[y2, x2]
        
        # Calculate the absolute difference
        if abs(int(pixel_value1) - int(pixel_value2)) < good_match_threshold:
            good_matches.append(((x1, y1), (x2, y2)))

    return good_matches

def main(image_path1, image_path2, num_random_points=100, downsample_factor=1, good_match_threshold=30):
    image1, image2 = load_and_prepare_images(image_path1, image_path2)

    if downsample_factor > 1:
        image1 = downsample_image(image1, downsample_factor)
        image2 = downsample_image(image2, downsample_factor)

    # Generate random points
    random_points1 = generate_random_points(image1, num_random_points)
    random_points2 = generate_random_points(image2, num_random_points)

    # Calculate good matches
    good_matches = calculate_good_matches(random_points1, random_points2, image1, image2, good_match_threshold)
    good_matches_count = len(good_matches)

    print(f"Good Matches Count: {good_matches_count}")

    # Visualize matches
    plt.figure(figsize=(12, 6))

    # Show Image 1 with random points and matches
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Image 1 with Random Points')
    plt.axis('off')
    for (x1, y1), (x2, y2) in good_matches:
        plt.scatter(x1, y1, color='red', s=10)
    
    # Show Image 2 with random points and matches
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title('Image 2 with Random Points')
    plt.axis('off')
    for (x1, y1), (x2, y2) in good_matches:
        plt.scatter(x2, y2, color='blue', s=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path1 = "/home/sristy/Desktop/Project-Based-Learning/Colon/10x/6_colon_10x.tif"
    image_path2 = "/home/sristy/Desktop/Project-Based-Learning/Colon/10x/10_colon_10x.tif"
    num_random_points = 200  
    downsample_factor = 2 
    main(image_path1, image_path2, num_random_points=num_random_points, downsample_factor=downsample_factor)
