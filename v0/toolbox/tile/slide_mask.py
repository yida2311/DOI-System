import os
import math
import cv2
import json
import openslide

from PIL import Image
import numpy as np
from xml.dom import minidom


def generate_std_mask(slide, tissue_mask_dir, anno_mask_dir, save_dir):
    tissue_mask_path = os.path.join(tissue_mask_dir, slide+'.png')
    anno_mask_path = os.path.join(anno_mask_dir, slide+'.png')
    print(anno_mask_path)
    tissue_mask = np.array(Image.open(tissue_mask_path).convert('1'))
    anno_mask = np.array(Image.open(anno_mask_path))
    mask = tissue_mask * anno_mask + tissue_mask
    mask = Image.fromarray(mask)

    mask.save(os.path.join(save_dir, slide+'.png'))


def generate_anno_mask(slide, scale, data_dir, save_dir):
    print(slide)
    wsi_path = os.path.join(data_dir, slide+'.svs')
    anno_path = os.path.join(data_dir, slide+ '.xml')

    wsi = openslide.OpenSlide(wsi_path)
    w, h = wsi.dimensions
    print(w, h)
    mask = np.zeros((h, w), dtype=np.uint8)

    regions, labels = get_regions(anno_path)
    for region, label in zip(regions, labels):
        region = region.reshape(-1, 1, 2)
        cv2.polylines(mask, np.int32([region]), True, label)
        cv2.fillPoly(mask, np.int32([region]), label)
    
    re_w = math.floor(w / scale); re_h = math.floor(h / scale)
    mask = cv2.resize(mask, (re_w, re_h))
    cv2.imwrite(os.path.join(save_dir, slide+'.png'), mask)


def get_regions(path):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(path)
    # The first region marked is always the tumour delineation
    annotations = xml.getElementsByTagName('Annotation')
    for annotation in annotations:
        regions = annotation.getElementsByTagName('Region')
        if regions is None or len(regions) == 0:
            continue
        elif regions[0].getAttribute('Text') == "":
            continue
        else:
            region_coord, region_labels = [], []
            for region in regions:
                vertices = region.getElementsByTagName('Vertex')
                attribute = region.getElementsByTagName("Attribute")
                if len(attribute) > 0:
                    r_label = attribute[0].attributes['Value'].value
                else:
                    r_label = int(region.getAttribute('Text')) + 1
                region_labels.append(r_label)

                # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = int(vertex.attributes['X'].value)
                    coords[i][1] = int(vertex.attributes['Y'].value)
                region_coord.append(coords)
            
    return region_coord, region_labels


if __name__ == '__main__':
    DATA_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/orig_data/'
    ANNO_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/mask_5x/anno_mask/'
    STD_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/mask_5x/std_mask/'
    TISSUE_MASK_DIR = '/media/ldy/7E1CA94545711AE6/OSCC/filter_5x/filtered_mask_sel/'
    SCALE = 8

    slide_list = os.listdir(TISSUE_MASK_DIR)
    slide_list = [c.split('.')[0] for c in slide_list]

    # for slide in slide_list:
    #     generate_anno_mask(slide, SCALE, DATA_DIR, ANNO_MASK_DIR)
    
    for slide in slide_list:
        generate_std_mask(slide, TISSUE_MASK_DIR, ANNO_MASK_DIR, STD_MASK_DIR)
