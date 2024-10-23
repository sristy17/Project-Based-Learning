import cv2
import numpy as np
import cProfile

def overlay_images_orb(img1, img2):
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    # Extract location of keypoints from good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Calculate matching percentage
    total_keypoints = len(kp1) + len(kp2)
    matching_keypoints = len(good_matches) * 2
    if total_keypoints > 0:
        matching_percentage = (matching_keypoints / total_keypoints) * 100
    else:
        matching_percentage = 0

    print(f"Number of matching keypoints (Inliers): {len(good_matches)}")
    print(f"Percentage of matching keypoints: {matching_percentage:.2f}%")

    if len(good_matches) >= 4:  # At least 4 matches needed to find homography
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp second image to match the first one
        warped_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

        # Create an overlaid image using weighted addition
        overlaid_image = cv2.addWeighted(img1, 0.7, warped_img2, 0.3, 0)
    else:
        print("Not enough good matches found to compute homography.")
        overlaid_image = img1  # Return the first image if not enough matches

    return overlaid_image

def main():
    img1 = cv2.imread("/home/sristy/Desktop/Project-Based-Learning/Pancreas/40x_png/35_Pancreas_40x.png")
    img2 = cv2.imread("/home/sristy/Desktop/Project-Based-Learning/Pancreas/40x_png/36_Pancreas_40x.png")

    if img1 is None or img2 is None:
        print("Error loading images")
        return

    # Call overlay function
    overlaid_image = overlay_images_orb(img1, img2)

    # Resize the output for better visualization
    resized_image = cv2.resize(overlaid_image, None, fx=0.5, fy=0.5)

    # Display the result
    cv2.imshow("Overlaid Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ensure the program runs only if it is the main script
if __name__ == '__main__':
    cProfile.run('main()')
