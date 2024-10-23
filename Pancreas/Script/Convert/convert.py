import cv2
import os


input_folder = 'C:/Image Processing/COLON/10x'
output_folder = 'C:/Image Processing/COLON/10x_png'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith(".tif"):
        
        input_path = os.path.join(input_folder, filename)
        
        
        tiff_image = cv2.imread(input_path)
        
        
        output_filename = f"{i+1}_colon_10x.png"  
        output_path = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_path, tiff_image)
        
        print(f"Converted {filename} to {output_filename}")

print("All images converted.")
