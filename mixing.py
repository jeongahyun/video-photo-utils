import cv2

image_a = cv2.imread("")
image_b = cv2.imread("")

for i in range(11):
    weight_a = 0.1 * i
    weight_b = 1 - weight_a

    result = image_a * weight_a + image_b * weight_b

    cv2.imwrite(f"", result)

