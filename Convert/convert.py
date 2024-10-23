import cv2
import os

input_folder = '/home/sristy/Desktop/Project-Based-Learning/Pancreas/40x data'
output_folder = '/home/sristy/Desktop/Project-Based-Learning/Pancreas/40x_png'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


filenames = sorted(os.listdir(input_folder))


for i, filename in enumerate(filenames):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_folder, filename)

    
        tiff_image = cv2.imread(input_path)

        output_filename = filename.replace('.tif', '.png')
        output_path = os.path.join(output_folder, output_filename)

        cv2.imwrite(output_path, tiff_image)

        print(f"Converted {filename} to {output_filename}")

print("All images converted.")
