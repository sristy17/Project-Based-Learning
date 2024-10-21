"""
1. Import deformed and reference image
2. Convert to greyscale
3. Initialize ORB object
4. Find key points and define them
5. Match keypoints (reference to deformed)
6. Error filtering - using RANSAC
7. Register 2 images(use homology)
"""
import cv2
import numpy as np

im1 = cv2.imread('/home/sristy/Desktop/ORB-Descriptor/ORB_Sristy/output/best-fit/best-fit1.jpg')#image to be registered
im2 = cv2.imread('/home/sristy/Desktop/ORB-Descriptor/ORB_Sristy/input/input_img.jpg')#reference image

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#MATCH DESCRIPTORS
matches = matcher.match(des1, des2, None)
matches = sorted(matches, key=lambda x:x.distance)

points1 = np.zeros((len(matches), 2), dtype = np.float32)
points2 = np.zeros((len(matches), 2), dtype = np.float32)
for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt 
    points2[i, :] = kp2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)#to fond the homography image

#using homography
height, width, channels = im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width, height))


img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None)
cv2.imshow("Matchpoints Image", img3)
cv2.imshow("Registered Image", im1Reg)
cv2.waitKey(0)