import os
import cv2
import numpy as np 

from .util import Time, open_image, open_image_np, np_to_pil, mask_rgb, mask_percent, mask_percentage_text, display_img
from .basic_filter import filter_green_channel, filter_grays, filter_red_pen, filter_green_pen, filter_blue_pen, filter_remove_small_holes, filter_remove_small_objects
#import basic_filter as filter


def color_filter(np_img, item, save_dir=None, save=False, display=False, hole_size=1000, object_size=600):
    rgb = np_img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_display(rgb, save_dir, item, save, False, 1, 'Original', 'rgb')

    mask_not_green = filter_green_channel(rgb)
    rgb_not_green = mask_rgb(rgb, mask_not_green)
    save_display(rgb_not_green, save_dir, item, save, False, 2, 'Not Green', 'rgb-not-green')

    mask_not_gray = filter_grays(rgb, tolerance=13)
    rgb_not_gray = mask_rgb(rgb, mask_not_gray)
    save_display(rgb_not_gray, save_dir, item, save, False, 3, 'Not Gray', 'rgb-not-gray')

    mask_not_red_pen = filter_red_pen(rgb)
    rgb_not_red_pen = mask_rgb(rgb, mask_not_red_pen)
    save_display(rgb_not_red_pen, save_dir, item, save, False, 4, 'Not Red Pen', 'rgb-not-red-pen')

    mask_not_green_pen = filter_green_pen(rgb)
    rgb_not_green_pen = mask_rgb(rgb, mask_not_green_pen)
    save_display(rgb_not_green_pen, save_dir, item, save, False, 5, 'Not Green Pen', 'rgb-not-green-pen')

    mask_not_blue_pen = filter_blue_pen(rgb)
    rgb_not_blue_pen = mask_rgb(rgb, mask_not_blue_pen)
    save_display(rgb_not_blue_pen, save_dir, item, save, False, 6, 'Not Blue Pen', 'rgb-not-blue-pen')


    mask_gray_green_pens = mask_not_gray & mask_not_green & mask_not_red_pen & mask_not_green_pen & mask_not_blue_pen
    rgb_gray_green_pens = mask_rgb(rgb, mask_gray_green_pens)
    save_display(rgb_gray_green_pens, save_dir, item, save, False, 7, "Not Gray, Not Green, No Pens",
               "rgb-no-gray-no-green-no-pens")
    
    mask_remove_holes = filter_remove_small_holes(mask_not_gray , min_size=hole_size, output_type="bool")
    rgb_remove_holes = mask_rgb(rgb, mask_remove_holes)
    save_display(rgb_remove_holes, save_dir, item, save, False, 8, 
                "Not Gray, Not Green, No Pens, \n Remove Small Holes",
               "rgb-not-green-not-gray-no-pens-remove-small-holes")

    mask_remove_obejcts = filter_remove_small_objects(mask_remove_holes, min_size=object_size, output_type="bool")
    rgb_remove_obejcts = mask_rgb(rgb, mask_remove_obejcts)
    save_display(rgb_remove_obejcts, save_dir, item, save, False, 9, 
                "Not Gray, Not Green, No Pens, \n Remove Small Objects",
               "rgb-not-green-not-gray-no-pens-remove-small-objects")
    
    n, label, stats, centrids = cv2.connectedComponentsWithStats(np.array(mask_remove_obejcts, dtype='uint8'), connectivity=8)
    index = np.argmax(stats[1:, 4])+1
    mask_remove_isolated = np.zeros_like(mask_remove_obejcts, dtype='bool')
    mask_remove_isolated[label==index] = 1
    rgb_remove_isolated = mask_rgb(rgb, mask_remove_isolated)

    return rgb_remove_isolated, mask_remove_isolated, rgb_remove_obejcts, mask_remove_obejcts


def save_display(np_img, dir, item, save, display, filter_num, display_text, file_text, 
                display_mask_percentage=True):
    mask_percentage = None
    if display_mask_percentage:
        mask_percentage = mask_percent(np_img)
        display_text = display_text + '\n(' + mask_percentage_text(mask_percentage) + ' masked)'
    if filter_num is None:
        display_text = item + '_' + display_text
    else:
        display_text = item + '_' + str(filter_num) + '_' + display_text
    
    if display:
        display_img(np_img, display_text)
    
    if save:
        file_text = item + '_' + str(filter_num) + '_' + file_text + '.png'
        t = Time()
        file_path = os.path.join(dir, file_text)
        pil_img = np_to_pil(np_img)
        pil_img.save(file_path)
        print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), file_path))


