import os
import numpy as np
import openslide
from PIL import Image, ImageChops
import math
import random

from store import *


def get_patch_info(shape, p_size, overlap):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    step_x = step_y = p_size - overlap
    n = math.ceil(x / step_x)
    m = math.ceil(y / step_y)
    step_x = (x - p_size) * 1.0 / (n - 1)
    step_y = (y - p_size) * 1.0 / (m - 1)

    return n, m, step_x, step_y


def sample_store_patches_svs(file,
                        file_dir,
                        file_mask_dir,
                        save_patch_dir,
                        patch_size,
                        level,
                        overlap,
                        filter_rate,
                        scale_factor=8,
                        sample_type='seg', # 'seg' or 'cls
                        save_mask_dir=None,
                        resize_factor=1,
                        rows_per_iter=1,
                        storage_format='png',
                        ):
    ''' Sample patches of specified size from .svs file.
        - file             name of whole slide image to sample from
        - file_dir              directory file is located in
        - file_mask_dir              directory mask file is located in 
        - save_patch_dir        directory patches is stored
        - patch_size            size of patches
        - level                 0 is lowest resolution; level_count - 1 is highest
        - overlap               pixels overlap on each side
        - filter_rate           rate of tissue
        - scale_factor          scale between wsi and mask
        - sample_type           sample type for task segmentation or classification
        - save_mask_dir         directory patches mask is stored
        - resize_factor         resize factor of sampled patches
        - storage_type          the patch storage option
        - rows_per_txn          how many patches to load into memory at once
                     
        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''
    file_name = file + '.svs'
    file_mask_name = file + '.png'
    tile_size = patch_size - 2 * overlap

    db_location = os.path.join(save_patch_dir, file)
    mask_location = os.path.join(save_mask_dir, file)
    if not os.path.exists(db_location):
        os.makedirs(db_location)
    if not os.path.exists(mask_location):
        os.makedirs(mask_location)

    slide = openslide.open_slide(os.path.join(file_dir, file_name))
    slide_mask = Image.open(os.path.join(file_mask_dir, file_mask_name))

    W, H = slide.dimensions
    # num of tiles in col and ver direction; cut step in col and ver
    col_tiles, ver_tiles, col_step, ver_step = get_patch_info([W, H], patch_size, overlap)
    generator = PatchGenerator(slide, slide_mask, patch_size, scale=scale_factor, resize_factor=resize_factor, sample_type=sample_type)
    slide_info = dict(size=(H, W), tiles=(ver_tiles, col_tiles), step=(ver_step, col_step))
    
    print("ver_tiles({}), col_tiles({})".format(ver_tiles, col_tiles))
    count = 0
    coord = [0, 0]
    for ver in range(ver_tiles):
        if ver < ver_tiles-1:
            coord[0] = int(ver * ver_step)
        else:
            coord[0] = int(H - patch_size)
        for col in range(col_tiles):
            print(ver, col)
            if col < col_tiles - 1:
                coord[1] = int(col * col_step)
            else:
                coord[1] = int(W - patch_size)
            
            tile_result = generator.get_patch_filtered(coord, filter_rate, level)
            if tile_result is None:
                continue
            filtered_tile, target = tile_result
            
            if sample_type == 'seg':
                patch_save_to_png_seg(filtered_tile, db_location, [ver, col], file)
                patch_save_to_png_seg(target, mask_location, [ver, col], file)
            else:
                patch_save_to_png_cls(filtered_tile, db_location, [ver, col], file, target)
            
            count += 1
            
    return slide_info, count


def sample_store_patches_svs_test(file,
                        file_dir,
                        file_mask_dir,
                        save_patch_dir,
                        patch_size,
                        level,
                        overlap,
                        filter_rate,
                        scale_factor=8,
                        sample_type='seg', # 'seg' or 'cls
                        resize_factor=1,
                        rows_per_iter=1,
                        storage_format='png',
                        ):
    ''' Sample patches of specified size from .svs file.
        - file             name of whole slide image to sample from
        - file_dir              directory file is located in
        - file_mask_dir              directory mask file is located in 
        - save_patch_dir        directory patches is stored
        - patch_size            size of patches
        - level                 0 is lowest resolution; level_count - 1 is highest
        - overlap               pixels overlap on each side
        - filter_rate           rate of tissue
        - scale_factor          scale between wsi and mask
        - sample_type           sample type for task segmentation or classification
        - resize_factor         resize factor of sampled patches
        - storage_type          the patch storage option
        - rows_per_txn          how many patches to load into memory at once
                     
        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''
    file_name = file + '.svs'
    file_mask_name = file + '.png'
    tile_size = patch_size - 2 * overlap

    db_location = os.path.join(save_patch_dir, file)
    if not os.path.exists(db_location):
        os.makedirs(db_location)

    slide = openslide.open_slide(os.path.join(file_dir, file_name))
    slide_mask = Image.open(os.path.join(file_mask_dir, file_mask_name))

    W, H = slide.dimensions
    # num of tiles in col and ver direction; cut step in col and ver
    col_tiles, ver_tiles, col_step, ver_step = get_patch_info([W, H], patch_size, overlap)
    generator = PatchGenerator(slide, slide_mask, patch_size, scale=scale_factor, resize_factor=resize_factor, sample_type=sample_type)
    slide_info = dict(size=(H, W), tiles=(ver_tiles, col_tiles), step=(ver_step, col_step))
    
    print("ver_tiles({}), col_tiles({})".format(ver_tiles, col_tiles))
    count = 0
    coord = [0, 0]
    for ver in range(ver_tiles):
        if ver < ver_tiles-1:
            coord[0] = int(ver * ver_step)
        else:
            coord[0] = int(H - patch_size)
        for col in range(col_tiles):
            print(ver, col)
            if col < col_tiles - 1:
                coord[1] = int(col * col_step)
            else:
                coord[1] = int(W - patch_size)
            
            tile_result = generator.get_patch_filtered(coord, filter_rate, level)
            if tile_result is None:
                continue
            filtered_tile, target = tile_result
            
            patch_save_to_png_seg(filtered_tile, db_location, [ver, col], file)
            count += 1
            
    return slide_info, count


class PatchGenerator(object):
    def __init__(self, slide, mask, patch_size, scale=8, resize_factor=1, is_filter=True, sample_type='seg'):
        self.slide = slide
        self.mask = mask 
        self.mask_size = mask.size  # w, h
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size_resize = self.patch_size // self.scale
        self.resize_factor = resize_factor
        self.is_filter = is_filter
        self.sample_type = sample_type

    def get_patch_filtered(self, coord, filter_rate, level):
        top = coord[0]; left = coord[1]
        top_resize = top // self.scale; left_resize = left // self.scale

        patch_mask = self.mask.crop((left_resize, top_resize, left_resize+self.patch_size_resize, top_resize+self.patch_size_resize))
        
        if self.is_filter and self._calculate_mask_rate(patch_mask) < filter_rate:
                print(self._calculate_mask_rate(patch_mask))
                return None
        
        patch = self.slide.read_region((left, top), level, (self.patch_size, self.patch_size))
        patch = patch.resize((self.patch_size//self.resize_factor, self.patch_size//self.resize_factor))
        if self.sample_type == 'seg':
            patch_mask = patch_mask.resize(patch.size)
            return patch, patch_mask
        else:
            target = self._mask_to_cls_label(patch_mask)
            return patch, target
    
    def _calculate_mask_rate(self, mask):
        """
        mask: PIL image
        """
        mask_bin = np.array(mask) > 0
        return mask_bin.sum() / mask_bin.size
    
    def _mask_to_cls_label(self, mask):
        mask = np.array(mask)
        hist = [0, 0, 0]
        for i in range(3):
            hist[i] = (mask==(i+1)).sum()
        return hist.index(max(hist)) + 1






