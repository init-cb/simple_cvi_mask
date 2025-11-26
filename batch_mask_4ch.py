'''

左室（LV）：100 green/ cyan
心肌（Myo）：120 red/megenta
右室（RV）：140 ?
左房（LA）：160  yellow
右房（RA）：180 blue

'''

import os

import cv2

PATH_ROOT = "//mnt/f/datasets/example/251120_example"

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


def connect_dotted_line(image_hsv, line_mask):
    size_k = 111
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size_k, size_k))
    # kernel = np.ones((size_k,size_k), np.uint8)
    mask_closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

    return mask_closed


def get_mask_from_png(img_path, DEBUG_DIR=None, colors=None):
    # ensure_dir(DEBUG_DIR)

    # 1. 读图并转 HSV
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RV
    mask_dark_blue_line = cv2.inRange(hsv, LOWER_DARK_BLUE, UPPER_DARK_BLUE)

    # RA
    mask_blue_line = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # myo
    mask_cyan_line = cv2.inRange(hsv, LOWER_CYAN, UPPER_CYAN)

    # LV
    mask_magenta_line = cv2.inRange(hsv, LOWER_MAGENTA, UPPER_MAGENTA)

    # LA
    mask_yellow_line = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    lines_full = mask_magenta_line + mask_yellow_line + mask_cyan_line + mask_blue_line + mask_dark_blue_line

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_lv_magenta_line.png"), mask_magenta_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_myo_cyan_line.png"), mask_cyan_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_la_yellow_raw.png"), mask_yellow_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_ra_blue_raw.png"), mask_blue_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_rv_dark_blue_line.png"), mask_dark_blue_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "01_1_lines_full.png"), lines_full)

    # mask_cyan_line = connect_dotted_line(hsv,mask_cyan_line)
    from dcm_nii_png_mask.circle_connect import connect_points_no_intersection

    mask_magenta_line = connect_points_no_intersection(line=mask_magenta_line)
    mask_cyan_line = connect_points_no_intersection(line=mask_cyan_line)
    mask_yellow_line = connect_points_no_intersection(line=mask_yellow_line)
    mask_blue_line = connect_points_no_intersection(line=mask_blue_line)
    mask_dark_blue_line = connect_points_no_intersection(line=mask_dark_blue_line)

    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_mask_magenta_line_connected.png"), mask_magenta_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_cyan_line_connected.png"), mask_cyan_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_mask_yellow_line_connected.png"), mask_yellow_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_mask_blue_line_connected.png"), mask_blue_line)
        cv2.imwrite(os.path.join(DEBUG_DIR, "06_mask_dark_blue_line_connected.png"), mask_dark_blue_line)

    # 1 darkblue; 2blue; 3 magenta; 4 cyan; 5 yellow
    fill_full = fill_region_from_line(
        mask_magenta_line + mask_yellow_line + mask_cyan_line + mask_blue_line + mask_dark_blue_line)

    fill_rv = fill_region_from_line(mask_dark_blue_line)

    fill_la = fill_region_from_line(mask_yellow_line)
    fill_ra = fill_region_from_line(mask_blue_line)

    fill_lv_myo = fill_full & (~fill_la) & (~fill_ra)

    fill_lv = fill_region_from_line(mask_magenta_line)
    fill_myo = fill_lv_myo & (~fill_lv) & (~fill_rv)


    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "07_fill_full.png"), fill_full)
        cv2.imwrite(os.path.join(DEBUG_DIR, "08_fill_la.png"), fill_la)
        cv2.imwrite(os.path.join(DEBUG_DIR, "09_fill_lv_myo.png"), fill_lv_myo)
        # cv2.imwrite(os.path.join(DEBUG_DIR, "fill_la_lv.png"),    fill_la_lv)
        cv2.imwrite(os.path.join(DEBUG_DIR, "10_fill_lv.png"), fill_lv)
        cv2.imwrite(os.path.join(DEBUG_DIR, "11_fill_myo.png"), fill_myo)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[(fill_lv > 0)] = 100
    mask[(fill_myo > 0)] = 120
    mask[(fill_la > 0)] = 160
    mask[(fill_ra > 0)] = 180
    mask[(fill_rv > 0)] = 140
    if DEBUG_DIR:
        cv2.imwrite(os.path.join(DEBUG_DIR, "12_mask.png"), mask)
    #
    return mask


if __name__ == '__main__':

    import pydicom
    import tqdm
    from PIL import Image

    debug = False
    if debug:
        path = '/path/to/your/debug_image.png'

        get_mask_from_png(path, DEBUG_DIR="/mnt/e/code/Other/utilities/dcm_nii_png_mask/debug")
    else:
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
