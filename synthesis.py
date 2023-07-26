import cv2
import numpy as np

target = cv2.imread("") 
source = cv2.imread("")
target_mask = cv2.imread("")
source_mask = cv2.imread("")

kernel = np.ones((3,3), np.uint8)
source_mask_dilation = cv2.dilate(source_mask, kernel, iterations=7)
cv2.imwrite("", source_mask_dilation)
# syn = cv2.copyTo(source, source_mask, target)
# syn = cv2.copyTo(source, target_mask, target)

# cv2.imwrite("", syn)
# cv2.imwrite("", syn)
