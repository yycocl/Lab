# Function: calculate the psnr
# Author: YuYao
# Time: 10/09/2019

import numpy as np
import cv2


def psnr(org_img, img):
    return 10*np.log10(255**2/mse(org_img, img))


def mse(org_img, img):
    # the way to calculate gray image and colorful image is different
    if len(img) == 3:
        diff_b = org_img[:][:][0] - img[:][:][0]
        diff_g = org_img[:][:][1] - img[:][:][1]
        diff_r = org_img[:][:][2] - img[:][:][2]
        diff_b = diff_b.flatten()
        diff_g = diff_g.flatten()
        diff_r = diff_r.flatten()
        result = np.mean(diff_b ** 2 + diff_g ** 2 + diff_r ** 2)/3
    else:
        diff = org_img - img
        diff = diff.flatten()
        result = np.mean(diff ** 2)
    return result


if __name__ == "__main__":
    org_path = "../Set5/image_SRF_2/img_003_SRF_2_HR.png"
    test_path = "../BiCubic/img_003_bicubic.png"
    org_img = cv2.imread(org_path)
    org_img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

    test_img = cv2.imread(test_path)
    test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    print(psnr(org_img_gray, test_img_gray))
