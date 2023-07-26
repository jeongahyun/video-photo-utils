import cv2
import struct
import pickle
import numpy as np
import os

########################
source_align_dir = ""
source_dir = ""
target_align_dir = ""
target_dir = ""
save_dir = ""
#########################

def get_landmark(img_path):
    with open(img_path, "rb") as f:
        data = f.read()

    length = len(data)
    chunks = []
    data_counter = 0
    while data_counter < length:
        chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter+2])
        data_counter += 2

        if chunk_m_l != 0xFF:
            raise ValueError("No Valid JPG info in data")

        chunk_name = None
        chunk_size = None
        chunk_data = None
        chunk_ex_data = None
        is_unk_chunk = False

        if chunk_m_h & 0xF0 == 0xD0:
            n = chunk_m_h & 0x0F

            if n >= 0 and n <= 7:
                chunk_name = "RST%d" % (n)
                chunk_size = 0
            elif n == 0x8:
                chunk_name = "SOI"
                chunk_size = 0
                if len(chunks) != 0:
                    raise Exception("")
            elif n == 0x9:
                chunk_name = "EOI"
                chunk_size = 0
            elif n == 0xA:
                chunk_name = "SOS"
            elif n == 0xB:
                chunk_name = "DQT"
            elif n == 0xD:
                chunk_name = "DRI"
                chunk_size = 2
            else:
                is_unk_chunk = True
        elif chunk_m_h & 0xF0 == 0xC0:
            n = chunk_m_h & 0x0F
            if n == 0:
                chunk_name = "SOF0"
            elif n == 2:
                chunk_name = "SOF2"
            elif n == 4:
                chunk_name = "DHT"
            else:
                is_unk_chunk = True
        elif chunk_m_h & 0xF0 == 0xE0:
            n = chunk_m_h & 0x0F
            chunk_name = "APP%d" % (n)
        else:
            is_unk_chunk = True

        if chunk_size == None: #variable size
            chunk_size, = struct.unpack (">H", data[data_counter:data_counter+2])
            chunk_size -= 2
            data_counter += 2

        if chunk_size > 0:
            chunk_data = data[data_counter:data_counter+chunk_size]
            data_counter += chunk_size

        if chunk_name == "SOS":
            c = data_counter
            while c < length and (data[c] != 0xFF or data[c+1] != 0xD9):
                c += 1

            chunk_ex_data = data[data_counter:c]
            data_counter = c

        chunks.append ({'name' : chunk_name,
                        'm_h' : chunk_m_h,
                        'data' : chunk_data,
                        'ex_data' : chunk_ex_data,
                        })

        dfl_dict = {}
        for chunk in chunks:
            if chunk['name'] == 'APP15':
                if type(chunk['data']) == bytes:
                    dfl_dict = pickle.loads(chunk['data'])
        
    landmarks = dfl_dict['source_landmarks']
    return landmarks

def draw_landmarks(img, landmarks):
    for lm in landmarks:
        img = cv2.circle(img, (int(lm[0]), int(lm[1])), 5, (255, 0, 0), -1)
    return img

def get_triangles(landmarks, x, y):
    rect = (0, 0, y, x)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks)
    subdiv.insert([(0, 0), (0, x-1), (y-1, 0), (y-1, x-1), (0, int(x/2)-1), (int(y/2)-1, 0), (int(y/2)-1, x-1), (y-1, int(x/2)-1)])
    tlist = subdiv.getTriangleList()
    triangles = []
    for t in tlist:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        triangles.append([pt1, pt2, pt3])
    return triangles

def hard_triangles(landmarks, x, y):
    triangles = []
    rect_0, rect_1, rect_2, rect_3, rect_4, rect_5, rect_6, rect_7 = (0, 0), (int(y/2)-1, 0), (y-1, 0), (0, int(x/2)-1), (y-1, int(x/2)-1), (0, x-1), (int(y/2)-1, x-1), (y-1, x-1)
    def lm(i):
        return (int(landmarks[i][0]), int(landmarks[i][1]))
    triangles.append([rect_0, lm(17), rect_1])
    triangles.append([rect_1, lm(17), lm(26)])
    triangles.append([rect_1, lm(26), rect_2])
    triangles.append([rect_0, rect_3, lm(17)])
    triangles.append([rect_2, lm(26), rect_4])
    triangles.append([rect_3, lm(0), lm(17)])
    triangles.append([lm(17), lm(0), lm(33)])
    triangles.append([lm(17), lm(33), lm(26)])
    triangles.append([lm(26), lm(33), lm(16)])
    triangles.append([lm(26), lm(16), rect_4])
    triangles.append([lm(0), rect_3, lm(4)])
    triangles.append([lm(0), lm(4), lm(48)])
    triangles.append([lm(0), lm(48), lm(33)])
    triangles.append([lm(33), lm(48), lm(54)])
    triangles.append([lm(33), lm(54), lm(16)])
    triangles.append([lm(16), lm(54), lm(12)])
    triangles.append([lm(16), lm(12), rect_4])
    triangles.append([lm(48), lm(4), lm(8)])
    triangles.append([lm(48), lm(8), lm(54)])
    triangles.append([lm(54), lm(8), lm(12)])
    triangles.append([rect_3, rect_5, lm(4)])
    triangles.append([lm(4), rect_5, lm(8)])
    triangles.append([lm(8), rect_5, rect_6])
    triangles.append([lm(8), rect_6, rect_7])
    triangles.append([lm(8), rect_7, lm(12)])
    triangles.append([lm(12), rect_7, rect_4])
    return triangles

def resize(dir_path, size):
    imgs = os.listdir(dir_path)
    for img in imgs:
        i = cv2.imread(f"{dir_path}/{img}")
        resized = cv2.resize(i, (size, size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{dir_path}/{img}", resized)

def draw_triangles(img, triangles):
    for tri in triangles:
        pt1 = tri[0]
        pt2 = tri[1]
        pt3 = tri[2]
        img = cv2.line(img, pt1, pt2, (0, 255, 0), 1)
        img = cv2.line(img, pt2, pt3, (0, 255, 0), 1)
        img = cv2.line(img, pt1, pt3, (0, 255, 0), 1)
    return img

def warp(img, tri1, tri2):
    matrix = cv2.getAffineTransform(np.float32(tri1), np.float32(tri2))
    warped = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_CUBIC)
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(tri2), (255, 255, 255))
    # warped = cv2.bitwise_and(warped, warped, mask=mask)
    # out = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    # out = out + warped
    return (warped, mask)

for i in range(len(os.listdir(source_dir))):
    source = cv2.imread(source_dir+f"/{str(i).zfill(5)}.png")
    target = cv2.imread(target_dir+f"/{str(i).zfill(5)}.png")
    source_lm = get_landmark(source_align_dir+f"/{str(i).zfill(5)}_0.jpg")
    target_lm = get_landmark(target_align_dir+f"/{str(i).zfill(5)}_0.jpg")
    source_tri = get_triangles(source_lm, source.shape[0], source.shape[1])
    target_tri = get_triangles(target_lm, target.shape[0], target.shape[1])

    source_tri_hard = hard_triangles(source_lm, source.shape[0], source.shape[1])
    target_tri_hard = hard_triangles(target_lm, target.shape[0], target.shape[1])

    pair = []
    for s, t in zip(source_tri_hard, target_tri_hard):
        pair.append(warp(source, s, t))
    out = source.copy()
    for p in pair:
        warped = cv2.bitwise_and(p[0], p[0], mask=p[1])
        out = cv2.bitwise_and(out, out, mask=cv2.bitwise_not(p[1]))
        out = out + warped
    cv2.imwrite(f"{save_dir}/{str(i).zfill(5)}.png", out)