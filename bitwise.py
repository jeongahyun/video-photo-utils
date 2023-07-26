import cv2

source = cv2.imread("") 
target = cv2.imread("")

XOR = cv2.bitwise_xor(source, target)
source_xor = cv2.bitwise_and(XOR, source)
target_xor = cv2.bitwise_and(XOR, target)

cv2.imwrite("", XOR)
cv2.imwrite("", source_xor)
cv2.imwrite("", target_xor)