
#     mask[(fill_lv>0)] = 100
#     mask[(fill_myo>0)] = 120
#     mask[(fill_la>0)] = 160
#     mask[(fill_ra>0)] = 180
#     # mask[(fill_rv>0)] = 140


import numpy as np


# HSV 颜色阈值（大致值，必要时根据你的图微调）
# 玫红
LOWER_MAGENTA = np.array([140, 150, 150], dtype=np.uint8)
UPPER_MAGENTA = np.array([179, 255, 255], dtype=np.uint8)

# 青绿
LOWER_CYAN = np.array([75, 150, 180], dtype=np.uint8)
UPPER_CYAN = np.array([85, 255, 255], dtype=np.uint8)

# 黄色（接近 FDFF48，适当收窄 Hue；S 下限偏高以避免白字）
LOWER_YELLOW = np.array([20, 210, 200], dtype=np.uint8)
UPPER_YELLOW = np.array([30, 255, 255], dtype=np.uint8)

MIN_AREA = 10

# blue
LOWER_BLUE = np.array([95, 150, 180], dtype=np.uint8)
UPPER_BLUE = np.array([100, 255, 255], dtype=np.uint8)

# dark blue
LOWER_DARK_BLUE = np.array([100, 220, 190], dtype=np.uint8)
UPPER_DARK_BLUE = np.array([120, 255, 255], dtype=np.uint8)
