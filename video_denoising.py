import cv2
import numpy as np

cap = cv2.VideoCapture('')

for i in range(3000):
    retval, frame = cap.read()
    if not(retval):
        break
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 3, 5)
    print(i)
    cv2.imwrite(f'{str(i).zfill(5)}.png', denoised)

