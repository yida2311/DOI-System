import math
import numpy as np 

from .util import Time, np_info, mask_percent
import skimage.morphology as sk_morphology


def filter_green_channel(np_img, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.

    Args:
        np_img: RGB image as a NumPy array.
        green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
        "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d" % (
            mask_percentage, overmask_thresh, green_thresh, new_green_thresh))
        gr_ch_mask = filter_green_channel(np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type)
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.

    Args:
        np_img: RGB image as a NumPy array.
        tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Grays", t.elapsed())
    return result


def filter_red_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out red pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
            filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
            filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
            filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
            filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
            filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
            filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
            filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
            filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Red Pen", t.elapsed())
    return result

def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool",
               display_np_info=False):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_lower_thresh: Red channel lower threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_upper_thresh: Blue channel upper threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Red", t.elapsed())
    return result


def filter_green_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out green pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
            filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
            filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
            filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
            filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
            filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
            filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
            filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
            filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
            filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
            filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
            filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
            filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
            filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
            filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Green Pen", t.elapsed())
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool",
                 display_np_info=False):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_lower_thresh: Green channel lower threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Green", t.elapsed())
    return result


def filter_blue_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out blue pen marks from a slide.

    Args:
        rgb: RGB image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array representing the mask.
    """
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
            filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
            filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
            filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
            filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
            filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
            filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
            filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
            filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
            filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
            filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
            filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    np_info(result, "Filter Blue Pen", t.elapsed())
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    Args:
        rgb: RGB image as a NumPy array.
        red_upper_thresh: Red channel upper threshold value.
        green_upper_thresh: Green channel upper threshold value.
        blue_lower_thresh: Blue channel lower threshold value.
        output_type: Type of array to return (bool, float, or uint8).
        display_np_info: If True, display NumPy array info and filter time.

    Returns:
        NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        np_info(result, "Filter Blue", t.elapsed())
    return result


def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8"):
    """
    Filter image to remove small holes less than a particular size.

    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Remove small holes below this size.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8).
    """
    t = Time()

    rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

    if output_type == "bool":
        pass
    elif output_type == "float":
        rem_sm = rem_sm.astype(float)
    else:
        rem_sm = rem_sm.astype("uint8") * 255

    np_info(rem_sm, "Remove Small Holes", t.elapsed())
    return rem_sm



def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.

    Args:
        np_img: Image as a NumPy array of type bool.
        min_size: Minimum size of small object to remove.
        avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
        overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
        output_type: Type of array to return (bool, float, or uint8).

    Returns:
        NumPy array (bool, float, or uint8).
    """
    t = Time()

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size / 2
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
        mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    np_info(np_img, "Remove Small Objs", t.elapsed())
    return np_img