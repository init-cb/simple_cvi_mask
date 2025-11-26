import pydicom
import os

import tqdm
from PIL import Image
import numpy as np
import cv2

from local_definition import *


def ensure_dir(d):
    if d is not None and len(d) > 0:
        os.makedirs(d, exist_ok=True)


def fill_region_from_line(line_mask, min_area=MIN_AREA):
    """
    根据一条闭合线的 mask（255 为线），找到其轮廓并向内填充。
    返回：填充后的区域 mask（uint8, 0/255）
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(line_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(line_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            # thickness=2 保持为线条，不填充
            cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
    return filled


def get_mask_from_png(img_path, DEBUG_DIR=None):
    # ensure_dir(DEBUG_DIR)

    # 1. 读图并转 HSV
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 按颜色取出三种线条的 mask
    mask_magenta_line = cv2.inRange(hsv, LOWER_MAGENTA, UPPER_MAGENTA)
    mask_cyan_line = cv2.inRange(hsv, LOWER_CYAN, UPPER_CYAN)
    mask_yellow_line = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_magenta_line.png"), mask_magenta_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "02_cyan_line.png"), mask_cyan_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "03_yellow_raw.png"), mask_yellow_line)

    fill_magenta = fill_region_from_line(mask_magenta_line)
    fill_cyan = fill_region_from_line(mask_cyan_line)
    fill_cyan_yellow = fill_region_from_line(mask_yellow_line + mask_cyan_line)

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "05_fill_magenta.png"), fill_magenta)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_fill_cyan.png"), fill_cyan)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_1_fill_cyan_yellow.png"), fill_cyan_yellow)

    lv = (fill_magenta > 0)

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "10_0_lv.png"), lv.astype(np.uint8) * 255)
    myo_lv = (fill_cyan > 0)
    Myo = myo_lv & (~lv)

    rv_myo_lv = (fill_cyan_yellow > 0)

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "10_1_Myo.png"), Myo.astype(np.uint8) * 255)

    rv = rv_myo_lv & (~myo_lv)
    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "10_2_rv.png"), rv.astype(np.uint8) * 255)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[lv] = 100
    mask[Myo] = 120
    mask[rv] = 140
    #     mask[(fill_lv>0)] = 100
    #     mask[(fill_myo>0)] = 120
    #     mask[(fill_la>0)] = 160
    #     mask[(fill_ra>0)] = 180
    #     # mask[(fill_rv>0)] = 140

    return mask


if __name__ == '__main__':

    path_dcm_series_single = "/path/to/your/image.dcm"
    basename = os.path.basename(path_dcm_series_single)
    output_dir_mask = "/mnt/e/code/Other/utilities/dcm_nii_png_mask/mask" + f"/{basename[:-4]}/"
    output_dir_dcm = "/mnt/e/code/Other/utilities/dcm_nii_png_mask/dcm" + f"/{basename[:-4]}/"
    output_dir_png = "/mnt/e/code/Other/utilities/dcm_nii_png_mask/png" + f"/{basename[:-4]}/"
    ds = pydicom.dcmread(path_dcm_series_single)

    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_dcm, exist_ok=True)
    os.makedirs(output_dir_png, exist_ok=True)

    frames = ds.pixel_array
    n_frames = frames.shape[0]

    print(f"检测到 {n_frames} 帧，开始拆分...")

    dcm_frames_png = []

    for i in tqdm.tqdm(range(n_frames)):
        new_ds = ds.copy()
        new_ds.PixelData = frames[i].tobytes()
        new_ds.NumberOfFrames = 1
        image_array = new_ds.pixel_array
        out_path = os.path.join(output_dir_dcm, f"frame_{i:04d}.dcm")
        out_path_png = os.path.join(output_dir_png, f"frame_{i:04d}.png")

        image = Image.fromarray(image_array)
        dcm_frames_png.append(image)
        image.save(out_path_png)
        new_ds.save_as(out_path)

    print("拆分完成！", os.path.basename(path_dcm_series_single))

    print(f"生成MASK...")

    for png in tqdm.tqdm(os.listdir(output_dir_png)):
        cv2.imwrite(os.path.join(output_dir_mask, f"{png}"), get_mask_from_png(os.path.join(output_dir_png, png)))

    print(f"done")
