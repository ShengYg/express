import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle
from progressbar import ProgressBar

"""
detect bar code
"""

def is_bar_code(w, h):
    area = w * h
    ratio = float(w) / h
    if area>2e4 and area<2e5 and ratio>1.5 and ratio<7 :
        return True
    return False


def pos_horizontal(img):

    im_scale = 1600 / float(img.shape[0])
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_shape = img.shape

    sobelx = cv2.Scharr(img, cv2.CV_16S, 1, 0)
    sobely = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(sobelx)
    absY = cv2.convertScaleAbs(sobely)
    abs_Y_sub_X = cv2.addWeighted(absX, -1, absY, 1, 0)

    blur = cv2.blur(abs_Y_sub_X, (7, 7))
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 3)
    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 3)

    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10,3)), iterations = 2)
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10,3)), iterations = 2)

    dst = thresh
    image, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    area_sort = [(i, area[i]) for i in range(len(area))]
    area_sort = sorted(area_sort, key=lambda x: x[1])[::-1]
    for i in range(3):
        ind = area_sort[i][0]
        x,y,w,h = cv2.boundingRect(contours[ind])
        if is_bar_code(h,w):
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.imwrite(path2 + name, color_img)
    return 0,0,0,0


    # for i in range(3):
    #     ind = area_sort[i][0]
    #     x,y,w,h = cv2.boundingRect(contours[ind])
    #     if is_bar_code(w, h):
    #         trans = cv2.transpose(img)
    #         if x1 + w1 > im_shape[1] / 3 * 2:
    #             flipped = cv2.flip(trans, 0)
    #         else:
    #             flipped = cv2.flip(trans, 1)
    #         return flipped
    # return img


def pos_vertical(img):
    """
    input image should be resized
    """
    im_shape = img.shape
    sobelx = cv2.Scharr(img, cv2.CV_16S, 1, 0)
    sobely = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(sobelx)
    absY = cv2.convertScaleAbs(sobely)
    abs_X_sub_y = cv2.addWeighted(absX, 1, absY, -1, 0)

    blur = cv2.blur(abs_X_sub_y, (7, 7))
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # v1
    # thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 1)
    # thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 1)
    # thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)), iterations = 2)
    # thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)), iterations = 2)

    # v2
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)), iterations = 3)
    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)), iterations = 3)

    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 2)
    thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3,10)), iterations = 2)

    dst = thresh
    image, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    area_sort = [(i, area[i]) for i in range(len(area))]
    area_sort = sorted(area_sort, key=lambda x: x[1])[::-1]
    for i in range(3):
        ind = area_sort[i][0]
        x,y,w,h = cv2.boundingRect(contours[ind])
        if is_bar_code(w, h):
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.imwrite(path2 + name, color_img)
    pos_horizontal(img)

    # for i in range(3):
    #     ind = area_sort[i][0]
    #     x,y,w,h = cv2.boundingRect(contours[ind])
    #     if is_bar_code(w, h):
    #         flipped = img
    #         if y1 + h1 > im_shape[0] / 4 * 3:
    #             flipped = cv2.flip(img, -1)
    #         return flipped
    # return pos_horizontal(img)


if __name__ == '__main__':

    path1 = '/home/sy/code/re_id/express/data/express/test/original/'
    path2 = '/home/sy/code/re_id/express/data/express/test/detected/'
    if not os.path.isdir(path2):
        os.makedirs(path2)
    nameall = sorted(os.listdir(path1))

    pbar = ProgressBar(maxval=len(nameall))
    pbar.start()
    i = 0
    for name in nameall:
        img = cv2.imread(path1 + name, 0)
        if img.shape[0] < 300:
            continue

        im_scale = 1600 / float(img.shape[1])
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        pos_vertical(img)

        # color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # cv2.rectangle(color_img,(x1,y1),(x1+w1,y1+h1),(0,0,255),5)
        # cv2.imwrite(path2 + name, color_img)

        # plt.figure()
        # ax1 = plt.subplot()
        # ax1.imshow(dst, cmap='gray')
        # ax1.add_patch(plt.Rectangle((x, y),
        #                           w, h, fill=False,
        #                           edgecolor='red', linewidth=1.5))
        # plt.show()

        i += 1
        pbar.update(i)
    pbar.finish()