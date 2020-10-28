# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:13:03 2020

@author: Carboxy
"""


import cv2 as cv2
import numpy as np


def get_convex_hull(mask_dir):
    '''
    获取mask中肿瘤最大连通区域的凸包
    Args:
        mask_dir: mask的路径。
    Returns:
        hull (array sized [N,2]): 凸包的点构成的数组。
    '''

    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE) # (H, W)
    mask = np.where(mask>35, 0, mask) # 肿瘤处灰度值为29左右，设阈值为35
    _, mask = cv2.threshold(mask,20,255,cv2.THRESH_BINARY) # 肿瘤处为255，其他为0

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 假设轮廓的点越多，所包含的面积越大，选取面积最大的绘制凸包
    contour_size = np.array([c.shape[0] for c in contours]) 
    main_contour_idx = np.argmax(contour_size)
    hull = cv2.convexHull(contours[main_contour_idx]) 
    hull = np.reshape(hull, (-1,2))

    return hull

def cross(hull, a, b, c):
    '''
    给定hull中点的序列a, b, c， 计算这三个点的围成的三角形面积，利用叉乘
    Args:
        hull (array sized [N,2]): 凸包的点构成的数组。
        a,b,c (int): 凸包中的点序列。
    Returns:
        S: 三角形的面积。
    '''
    y0 = hull[a,0]
    x0 = hull[a,1]
    y1 = hull[b,0]
    x1 = hull[b,1]
    y2 = hull[c,0]
    x2 = hull[c,1]    
    S = abs((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0))
    return S   

def dist2(hull, a, b):
    '''
    给定hull中的两个点，计算它们距离的平方
    '''
    y0 = hull[a,0]
    x0 = hull[a,1]
    y1 = hull[b,0]
    x1 = hull[b,1]
    return (y0-y1)**2 + (x0-x1)**2

def get_convex_hull_diameter(hull):
    '''
    获取凸包直径和对应的两个点，使用旋转卡壳法
    Args:
        hull (array sized [N,2]): 凸包的点构成的数组。
    Returns:
        dia: 凸包的直径。
    '''

    q = 1
    dia = 0
    N = len(hull)
    hull = np.vstack((hull, hull[0, :]))
    q_dia = 0
    p_dia = 0

    for p in range(N):
        while (cross(hull, p+1, q+1, p) > cross(hull, p+1, q, p)):
            q = (q+1)%N
        dist2_1 = dist2(hull, p, q)
        dist2_2 = dist2(hull, p+1, q+1)
        if max(dist2_1, dist2_2) > dia:
            dia = max(dist2_1, dist2_2)
            if dist2_1> dist2_1:
                p_dia = p
                q_dia = q
            else:
                p_dia = p+1
                q_dia = q+1
    
    return hull[p_dia,:], hull[q_dia, :], np.sqrt(dist2(hull, p_dia, q_dia))

def get_tumor_diameter(mask_dir):
    hull = get_convex_hull(mask_dir)
    return get_convex_hull_diameter(hull)

if __name__ == '__main__':
    hull = get_convex_hull('Masks/_20190718210428.png')
    p, q, dia = get_convex_hull_diameter(hull)
    print(p, ' ', q, ' ', dia)

    mask = cv2.imread('Masks/binary_contours.png')
    mask_dia = cv2.line(mask, (p[0], p[1]), (q[0], q[1]), (0,255,0), thickness=3)
    cv2.imwrite('Masks/binary_dia.png', mask_dia)