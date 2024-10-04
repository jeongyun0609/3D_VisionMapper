import os
import sys
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', required=True,
                    default="./14bit_image_path",
                    help="Thermal 14bit image path")   
parser.add_argument('-d', '--dst', required=True,
                    default="./8bit_image_path",
                    help="Thermal 8bit image path")                                                    
args = parser.parse_args()

# How to use:
# python3 convert_thermal_14bit_to_8bit.py -s=14bit_img.png -d=8bit_img.png

raw_T_image = cv2.imread(args.src, cv2.IMREAD_UNCHANGED)

h, w = raw_T_image.shape
min_thresh = np.min(raw_T_image)
max_thresh = np.max(raw_T_image)

for i in range(h):
    for j in range(w):
        if (raw_T_image[i][j]) > max_thresh:
            raw_T_image[i][j] = max_thresh

        elif (raw_T_image[i][j]) < min_thresh:
            raw_T_image[i][j] = min_thresh

normalized_array = (raw_T_image - min_thresh) / (max_thresh - min_thresh)
norm_img = (normalized_array * 255).astype(np.uint8)

cv2.imwrite(args.dst, norm_img)
