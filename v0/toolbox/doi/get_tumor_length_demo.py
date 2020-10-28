# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:13:03 2020

@author: Carboxy
"""


import cv2 as cv2
import numpy as np

#mask = cv2.imread('Masks/_20190718210428.png') # (H, W, 3)
mask = cv2.imread('Masks/_20190718210428.png', cv2.IMREAD_GRAYSCALE) # (H, W)
mask = np.where(mask>35, 0, mask) # 肿瘤处灰度值为29左右
_, mask = cv2.threshold(mask,20,255,cv2.THRESH_BINARY)
#cv2.imwrite('Masks/binary.png',mask)

contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_size = np.array([c.shape[0] for c in contours])
main_contour_idx = np.argmax(contour_size)
H, W = mask.shape
white_image = np.zeros((H, W, 3)).astype(np.uint8)

contours_image = cv2.drawContours(white_image, contours, main_contour_idx, (0, 255, 255), 2)
hull = cv2.convexHull(contours[main_contour_idx])
hull_image = cv2.drawContours(contours_image, [hull], 0, (255, 0, 255), 2)
#cv2.imwrite('Masks/top.png',top)
cv2.imwrite('Masks/binary_contours.png',hull_image)
a = 1



#print(w,h)
#mask_resize = cv2.resize(mask, (int(w/15),int(h/15)),interpolation=cv2.INTER_CUBIC)
#mask1 = np.where(mask_resize>100, mask_resize, 0)
#mask_reszie2 = np.where(mask_resize1<85, mask_resize1, 0)
#_, mask1 = cv2.threshold(mask,35,255,cv2.THRESH_BINARY)
#cv2.imwrite('Masks/gray_35.png',mask2)
# cv2.imshow('image', mask)
# cv2.waitKey (0) 
# cv2.destroyAllWindows()



