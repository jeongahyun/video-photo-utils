import cv2

img1 = cv2.imread("") 
img2 = cv2.imread("")

blended = img1 * 0.5 + img2 * 0.5

cv2.imwrite("", blended)