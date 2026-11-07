"""
エッジを求めて、拡大領域と縮小領域を出すコード
"""

import sys
import os
import glob
import numpy as np
import cv2

import os

# 確認用画像の保存先（make_expand_region.py と同じ場所の region_debug/ フォルダ）
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "region_debug")


def to_uint8(im):
    """
    cv2.imwrite で見える形（0〜255 の uint8）に直す。
      - 0〜255 の uint8 画像はそのまま
      - 0/1 の二値画像や float のエッジ強度は、最小〜最大を 0〜255 に引き伸ばす
        （そのまま保存すると真っ黒に見えるため。値の大きさは見た目用に変わる）
    """
    im = np.asarray(im)
    if im.dtype == np.uint8 and im.max() > 1:
        return im
    im = im.astype(np.float32)
    lo, hi = im.min(), im.max()
    if hi - lo < 1e-12:                      # 全画素が同じ値
        return np.zeros(im.shape, np.uint8)
    return ((im - lo) / (hi - lo) * 255).astype(np.uint8)


def save_debug_images(name, **images):
    """images の各画像を DEBUG_DIR/{name}_{キー名}.png として保存する"""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for key, im in images.items():
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_{key}.png"), to_uint8(im))

def get_expand_region(img, name, show_img=False):

    img_dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    img_dx = cv2.cvtColor(img_dx, cv2.COLOR_BGR2GRAY)
    img_dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    img_dy = cv2.cvtColor(img_dy, cv2.COLOR_BGR2GRAY)

    img_edge = (np.abs(img_dx) + np.abs(img_dy))/1284

    #img_edge = max_pooling_2x2(img_edge)
    #img_edge = max_pooling_2x2(img_edge)
    #img_edge = max_pooling_2x2(img_edge)
    #img_edge = cv2.resize(img_edge, (64, 36), interpolation=cv2.INTER_AREA) * 3

    _, img_th = cv2.threshold(img_edge, 0.25, 1.0, cv2.THRESH_BINARY)

    #img_edge = np.clip(img_edge / np.max(img_edge) * 255, 0, None).astype(np.uint8)
    #img_th = cv2.adaptiveThreshold(img_edge, 255,
    #                               cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 31, C=-40,)
    
    # クロージング処理を実行
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    #img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_th = cv2.dilate(img_th, kernel, iterations=2)
    img_th = cv2.erode(img_th, kernel, iterations=1)

    if show_img:
        save_debug_images(name, img=img, edge=img_edge, thresh=img_th)

    return img_th