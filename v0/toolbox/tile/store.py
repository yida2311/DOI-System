import h5py 
import os
from PIL import Image
import numpy as np
import csv

###########################################################################
#                Option 1: save patches to png                           #
###########################################################################

### Patch(es) saver for segmentation ###

def patch_save_to_png_seg(patch, db_location, coord, file_name):
    """PIL Image saver for segmentation"""
    fname = file_name + "_" + str(coord[0]) + "_" + str(coord[1]) + "_.png"
    patch.save(os.path.join(db_location, fname))


def patches_save_to_png_seg(patches, db_location, coords, file_name):
    """ Saves numpy patches to .png files (full resolution). 
        Meta data is saved in the file name.
        - db_location       folder to save images in
        - patches           PIL images
        - coords            x, y tile coordinates
        - file_name         original source WSI name
    """
    for i, patch in enumerate(patches):
        # Construct the new PNG filename
        patch_fname = file_name + "_" + str(coords[i][0]) + "_" + str(coords[i][1]) + "_.png"
        # Save the image.
        patch.save(os.path.join(db_location, patch_fname))

### Patch(es) saver for classification ###

def patch_save_to_png_cls(patch, db_location, coord, file_name, target):
    """PIL Image saver for classification"""
    fname = file_name + "_" + str(coord[0]) + "_" + str(coord[1]) + "_" + str(target) + "_.png"
    patch.save(os.path.join(db_location, fname))


def patches_save_to_png_cls(patches, db_location, coords, file_name, targets):
    for i, patch in enumerate(patches):
        patch_fname = file_name + "_" + str(coords[i][0]) + "_" + str(coords[i][1]) + "_" + str(target) + "_.png"
        patch.save(os.path.join(db_location, patch_fname))


###########################################################################
#                Option 2: store to HDF5 files                            #
###########################################################################

def save_to_hdf5(db_location, patches, coords, file_name, is_csv=True):
    """ Saves the numpy arrays to HDF5 files. All patches from a single WSI will be saved
        to the same HDF5 file, regardless of the transaction size specified by rows_per_txn,
        because this is the most efficient way to use HDF5 datasets.
        - db_location       folder to save images in
        - patches           numpy images
        - coords            x, y tile coordinates
        - file_name         original source WSI name
    """
    # Save patches into hdf5 file.
    file    = h5py.File(os.path.join(db_location, file_name + '.h5'),'w')
    dataset = file.create_dataset('t', np.shape(patches), h5py.h5t.STD_I32BE, data=patches)

    # Save all label meta into a csv file.
    if is_csv:
        with open(os.path.join(db_location, file_name + '.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(patches)):
                writer.writerow([coords[i][0], coords[i][1]])
