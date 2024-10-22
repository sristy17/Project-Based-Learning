import cv2
import numpy as np
import cProfile

def overlay_images_sift(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    total_keypoints = len(kp1) + len(kp2)
    matching_keypoints = len(good_matches) * 2
    matching_percentage = (matching_keypoints / total_keypoints) * 100

    print(f"Number of matching keypoints (Inliers): {len(good_matches)}")
    print(f"Numbers of points not matched (Outliers): {100 - len(good_matches)}")
    print(f"Percentage of matching keypoints: {matching_percentage:.2f}%")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    warped_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(img2, dtype=np.uint8), M, (img1.shape[1], img1.shape[0]))

    overlaid_image = cv2.addWeighted(img1, 0.7, warped_img2, 0.3, 0)

    return overlaid_image

def main():
    img1 = cv2.imread("C:/PBL_FINAL/Project-Based-Learning/Pancreas/10x data/1_Pancreas_10x.tif")
    img2 = cv2.imread("C:/PBL_FINAL/Project-Based-Learning/Pancreas/10x data/3_Pancreas_10x.tif")

    overlaid_image = overlay_images_sift(img1, img2)
    resized_image = cv2.resize(overlaid_image, None, fx=0.5, fy=0.5)

    cv2.imshow("Overlaid Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cProfile.run('main()')
