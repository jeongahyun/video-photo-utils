import cv2
import json
import numpy as np
import os

#===================================================================
# 프레임 저장된 폴더 경로, 변환된 프레임 저장할 폴더 경로 입력

FRAME_DIR = ''
SAVE_DIR = ''

#====================================================================

for i in range(len(os.listdir(FRAME_DIR))):
    if os.path.isfile(f'{FRAME_DIR}/{str(i).zfill(5)}.json'):
        with open(f'{FRAME_DIR}/{str(i).zfill(5)}.json') as f:
            label = json.load(f)
            neck_pts = np.array(label['shapes'][0]['points']).astype(np.int32)

    img = cv2.imread(f'{FRAME_DIR}/{str(i).zfill(5)}.png')
    black = np.zeros(img.shape, np.uint8)
    mask = cv2.fillPoly(black, [neck_pts], (255, 255, 255))
    mask_reverse = cv2.bitwise_not(mask)
    img_blur = cv2.GaussianBlur(img, (7,7), 0)

    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = np.clip(h-2, 0, 180)
    hsv = np.stack([h, s, v], axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = cv2.copyTo(img, mask_reverse, bgr)
    print(i)
    cv2.imwrite(f'{SAVE_DIR}/{str(i).zfill(5)}.png', out)
