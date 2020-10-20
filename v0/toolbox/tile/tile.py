"""
Main subslide generator manager, Cutter which keep track of a collection of SVS images, and allows for
subslide sampling, storing and accessing.

Author: yida2311

"""
import itertools
import math
import sys
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
#import openslide
import json
import pandas as pd

from .util import start_timer, end_timer
#from patch_reader_svs import sample_store_patches_svs, sample_store_patches_svs_test
from .patch_reader_png import sample_store_patches_png, sample_store_patches_png_test

STORAGE_TYPES = ['png', 'hdf5']


class Cutter(object):
    def __init__(self, 
                slide_list,
                file_dir,
                file_mask_dir,
                save_patch_dir,
                scale_factor=8,
                sample_type='seg',
                #save_mask_dir=None,
                storage_type='png',
                ):
        """
        Params:
            slide_list: list of slide names
            file_dir: SVS file path
            file_mask_dir: tissue mask path
            save_patch_dir: save file path
            save_mask_dir: target file path
            storage_type: expecting 'png', 'hdf5', 'npy'
        """
        if storage_type not in STORAGE_TYPES:
            print("[subslide error]: storage type not recognised; expecting one of", STORAGE_TYPES)
            return
        
        self.storage_type = storage_type
        self.files = slide_list
        self.file_dir = file_dir
        self.file_mask_dir = file_mask_dir
        self.save_patch_dir = save_patch_dir
        #self.save_mask_dir = save_mask_dir
        self.num_files = len(self.files)
        self.sample_type = sample_type
        self.scale_factor = scale_factor

        self.is_anno = True

        print("======================================================")
        print("Storage type:              ", self.storage_type)
        print("Images directory:          ", self.file_dir)
        #print("Mask directory:            ", self.file_mask_dir)
        print("Data store directory:      ", self.save_patch_dir)
        #print("Target store directory:    ", self.save_mask_dir)
        print("Images found:              ", self.num_files)
        print("======================================================")
    
    # def sample_and_store_patches_svs(self,
    #                                 patch_size,
    #                                 level,
    #                                 overlap,
    #                                 filter_rate,
    #                                 save_name='tile_info.json',
    #                                 resize_factor=1,
    #                                 rows_per_iter=1
    #                                 ):
    #     """ Samples patches from all whole slide images in the dataset and stores them in the
    #         specified format.
    #         - patch_size        the patch size in pixels to sample
    #         - level             the tile level to sample at
    #         - overlap           pixel overlap of patches
    #         - rows_per_txn      how many rows in the WSI to sample (save in memory) before saving to disk
    #                             a smaller number will use less RAM; a bigger number is slightly more
    #                             efficient but will use more RAM.
    #     """
    #     start_time = start_timer()
    #     total_num = 0
    #     info = {}
    #     for file in tqdm(self.files):
    #         print(file, end=" ")
    #         file_info, patches_num = sample_store_patches_svs(file,
    #                                                         self.file_dir,
    #                                                         self.file_mask_dir,
    #                                                         self.save_patch_dir,
    #                                                         patch_size,
    #                                                         level,
    #                                                         overlap,
    #                                                         filter_rate,
    #                                                         scale_factor=self.scale_factor,
    #                                                         sample_type=self.sample_type,
    #                                                         #save_mask_dir=self.save_mask_dir,
    #                                                         resize_factor=resize_factor,
    #                                                         rows_per_iter=rows_per_iter,
    #                                                         storage_format=self.storage_type)
    #         info[file] = file_info
    #         total_num += patches_num
        
    #     with open(os.path.join(self.save_patch_dir, save_name), 'w') as f:
    #         json.dump(info, f)
            
    #     print("")
    #     print("============ Patches Dataset Stats ===========")
    #     print("Total patches sampled:                    ", total_num)
    #     print("Patches saved to:                         ", self.save_patch_dir)
    #     print("")
    #     end_timer(start_time) 
    

    def sample_and_store_patches_png(self,
                                    patch_size,
                                    overlap,
                                    filter_rate,
                                    #save_name='tile_info.json',
                                    resize_factor=1,
                                    rows_per_iter=1
                                    ):
        """ Samples patches from all whole slide images in the dataset and stores them in the
            specified format.
            - patch_size        the patch size in pixels to sample
            - overlap           pixel overlap of patches
            - rows_per_txn      how many rows in the WSI to sample (save in memory) before saving to disk
                                a smaller number will use less RAM; a bigger number is slightly more
                                efficient but will use more RAM.
        """
        start_time = start_timer()
        total_num = 0
        info = {}
        target_info = []
        for file in self.files:
            print(file, end="  k ")
            #file_info, patches_num, target_info_slide = sample_store_patches_png(file,
            file_info, patches_num = sample_store_patches_png(file,
                                                            self.file_dir,
                                                            self.file_mask_dir,
                                                            self.save_patch_dir,
                                                            patch_size,
                                                            overlap,
                                                            filter_rate,
                                                            sample_type=self.sample_type,
                                                            #save_mask_dir=self.save_mask_dir,
                                                            resize_factor=resize_factor,
                                                            rows_per_iter=rows_per_iter,
                                                            storage_format=self.storage_type)
            info[file] = file_info
            total_num += patches_num
            #target_info += target_info_slide
            # print(target_info)
        # with open(os.path.join(self.save_patch_dir, save_name), 'w') as f:
        #     json.dump(info, f)
        
        #df = pd.DataFrame(target_info)
        # name = self.save_patch_dir.split('/')[-1]
        # print(name)
        # df.to_csv(os.path.join(self.save_patch_dir, name+'.csv'), columns=['image_id', 'slide_id', 'row', 'col', 'ratio', 'new_ratio', 'target'])

        print("")
        print("============ Patches Dataset Stats ===========")
        print("Total patches sampled:                    ", total_num)
        print("Patches saved to:                         ", self.save_patch_dir)
        print("")
        end_timer(start_time) 
        return file_info
    

    # def retrive_tile_dimensions(self, file_name, patch_size, overlap):
    #     """ For a given whole slide image in the dataset, retrieve the available tile dimensions.
    #         - file_name         the whole slide image filename
    #         - patch_size        patch size in pixels
        
    #         Returns:
    #         - level count
    #         - level tiles
    #         - level dimensions
    #     """
    #     tile_size = patch_size - 2*overlap
    #     assert tile_size > 0
    #     slide = openslide.open_slide(self.file_dir, file_name)
    #     tiles = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap)
    #     return tiles.level_count, tiles.level_tiles, tiles_level_dimensions
    
    def get_patches_from_file(self, file_name, verbose=False):
        """ Fetches the patches from one file, depending on storage method. 
        """
        if self.storage_type == 'png':
            return self.__get_patches_from_png(file_name[:-4], verbose)
        elif self.storage_type == 'hdf5':
            return self.__get_patches_from_hdf5(file_name[:-4], verbose)
        else:
            raise ValueError('Wrong storage_types: {}'.format(storage_type))


    ###########################################################################
    #           General class variable access functions                       #
    ###########################################################################
    def set_slide_list(self, slide_list):
        self._files = slide_list
        self.num_files = len(slide_list)

    def set_is_annotation(self, is_anno):
        self.is_anno = is_anno
    

    ###########################################################################
    #                PNG-specific helper functions                           #
    ###########################################################################  
    def __get_patches_from_png(self, wsi_name, verbose=False):
        """ Loads all the PNG patch images from disk. Note that this function does NOT 
            distinguish between other PNG images that may be in the directory; everything
            will be loaded.
        """
        # Get all files matching the WSI file name and correct file type.
        patch_files = np.array(
            [file for file in listdir(self.save_dir) 
            if isfile(join(self.save_dir, file)) and '.png' in file and wsi_name in file])

        patches, targets, coords = [], [], [], []
        for f in patch_files:
            patches.append(np.array(Image.open(self.save_dir + f), dtype=np.uint8))
            targets.append(np.array(Image.open(self.target_dir + f), dtype=np.uint8))
            f_ = f.split('_')
            coords.append([int(f_[1]), int(f_[2])])

        if verbose:
            print("[subslide] loaded", len(patches), "patches from", wsi_name)

        return patches,targets, coords

    ###########################################################################
    #                HDF5-specific helper functions                           #
    ###########################################################################  
    def __get_patches_from_hdf5(self, file_name, verbose=False):
        """ Loads the numpy patches from HDF5 files.
        """
        patches, targets, coords = [], [], [], []
        # Now load the images from H5 file.
        file = h5py.File(self.save_dir + file_name + ".h5",'r+')
        dataset = file['/' + 't']
        new_patches = np.array(dataset).astype('uint8')
        for patch in new_patches:
            patches.append(patch)
        file.close()

        file_target = h5py.File(self.target_dir + file_name + ".h5",'r+')
        dataset = file['/' + 't']
        new_patches = np.array(dataset).astype('uint8')
        for patch in new_patches:
            targets.append(patch)
        file_target.close()

        # Load the corresponding meta.
        with open(self.save_dir + file_name + ".csv", newline='') as metafile:
            reader = csv.reader(metafile, delimiter=' ', quotechar='|')
            for row in reader:
                coords.append([int(row[0]), int(row[1])])
        
        if verbose:
            print("[subslide] loaded from", file_name, ".h5 file", np.shape(patches))

        return patches, targets, coords



class Cutter_test(object):
    def __init__(self, 
                slide_list,
                file_dir,
                file_mask_dir,
                save_patch_dir,
                scale_factor=8,
                storage_type='png',
                ):
        """
        Params:
            slide_list: list of slide names
            file_dir: SVS file path
            file_mask_dir: tissue mask path
            save_patch_dir: save file path
            storage_type: expecting 'png', 'hdf5', 'npy'
        """
        if storage_type not in STORAGE_TYPES:
            print("[subslide error]: storage type not recognised; expecting one of", STORAGE_TYPES)
            return
        
        self.storage_type = storage_type
        self.files = slide_list
        self.file_dir = file_dir
        self.file_mask_dir = file_mask_dir
        self.save_patch_dir = save_patch_dir
        self.save_mask_dir = save_mask_dir
        self.num_files = len(self.files)
        self.sample_type = sample_type
        self.scale_factor = scale_factor

        print("======================================================")
        print("Storage type:              ", self.storage_type)
        print("Images directory:          ", self.file_dir)
        print("Mask directory:            ", self.file_mask_dir)
        print("Data store directory:      ", self.save_patch_dir)
        print("Images found:              ", self.num_files)
        print("======================================================")
    
    def sample_and_store_patches_svs(self,
                                    patch_size,
                                    level,
                                    overlap,
                                    filter_rate,
                                    resize_factor=1,
                                    rows_per_iter=1,
                                    ):
        """ Samples patches from all whole slide images in the dataset and stores them in the
            specified format.
            - patch_size        the patch size in pixels to sample
            - level             the tile level to sample at
            - overlap           pixel overlap of patches
            - rows_per_txn      how many rows in the WSI to sample (save in memory) before saving to disk
                                a smaller number will use less RAM; a bigger number is slightly more
                                efficient but will use more RAM.
        """
        start_time = start_timer()
        total_num = 0
        info = {}
        for file in tqdm(self.files):
            print(file, end=" ")
            file_info, patches_num = sample_store_patches_svs_test(file,
                                                                self.file_dir,
                                                                self.file_mask_dir,
                                                                self.save_patch_dir,
                                                                patch_size,
                                                                level,
                                                                overlap,
                                                                filter_rate,
                                                                scale_factor=self.scale_factor,
                                                                sample_type=self.sample_type,
                                                                resize_factor=resize_factor,
                                                                rows_per_iter=rows_per_iter,
                                                                storage_format=self.storage_type)
            total_num += patches_num
            info[file] = file_info
        with open(os.path.join(self.save_patch_dir, 'subslide_info.json'), 'w') as f:
            json.dump(info, f)
        print("")
        print("============ Patches Dataset Stats ===========")
        print("Total patches sampled:                    ", total_num)
        print("Patches saved to:                         ", self.save_dir)
        print("")
        end_timer(start_time) 
    

    def sample_and_store_patches_png(self,
                                    patch_size,
                                    overlap,
                                    filter_rate,
                                    resize_factor=1,
                                    rows_per_iter=1,
                                    ):
        """ Samples patches from all whole slide images in the dataset and stores them in the
            specified format.
            - patch_size        the patch size in pixels to sample
            - overlap           pixel overlap of patches
            - rows_per_txn      how many rows in the WSI to sample (save in memory) before saving to disk
                                a smaller number will use less RAM; a bigger number is slightly more
                                efficient but will use more RAM.
        """
        start_time = start_timer()
        total_num = 0
        info = {}
        for file in tqdm(self.files):
            print(file, end=" ")
            file_info, patches_num = sample_store_patches_png_test(file,
                                                                self.file_dir,
                                                                self.file_mask_dir,
                                                                self.save_patch_dir,
                                                                patch_size,
                                                                overlap,
                                                                filter_rate,
                                                                sample_type=self.sample_type,
                                                                resize_factor=resize_factor,
                                                                rows_per_iter=rows_per_iter,
                                                                storage_format=self.storage_type)
            total_num += patches_num
            info[file] = file_info
        with open(os.path.join(self.save_patch_dir, 'subslide_info.json'), 'w') as f:
            json.dump(info, f)
        print("")
        print("============ Patches Dataset Stats ===========")
        print("Total patches sampled:                    ", total_num)
        print("Patches saved to:                         ", self.save_dir)
        print("")
        end_timer(start_time) 
    

    # def retrive_tile_dimensions(self, file_name, patch_size, overlap):
    #     """ For a given whole slide image in the dataset, retrieve the available tile dimensions.
    #         - file_name         the whole slide image filename
    #         - patch_size        patch size in pixels
        
    #         Returns:
    #         - level count
    #         - level tiles
    #         - level dimensions
    #     """
    #     tile_size = patch_size - 2*overlap
    #     assert tile_size > 0
    #     slide = openslide.open_slide(self.file_dir, file_name)
    #     tiles = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap)
    #     return tiles.level_count, tiles.level_tiles, tiles_level_dimensions
    
    def get_patches_from_file(self, file_name, verbose=False):
        """ Fetches the patches from one file, depending on storage method. 
        """
        if self.storage_type == 'png':
            return self.__get_patches_from_png(file_name[:-4], verbose)
        elif self.storage_type == 'hdf5':
            return self.__get_patches_from_hdf5(file_name[:-4], verbose)
        else:
            raise ValueError('Wrong storage_types: {}'.format(storage_type))


    ###########################################################################
    #           General class variable access functions                       #
    ###########################################################################
    def set_slide_list(self, slide_list):
        self._files = slide_list
        self.num_files = len(slide_list)

    def set_is_annotation(self, is_anno):
        self.is_anno = is_anno
    

    ###########################################################################
    #                PNG-specific helper functions                           #
    ###########################################################################  
    def __get_patches_from_png(self, wsi_name, verbose=False):
        """ Loads all the PNG patch images from disk. Note that this function does NOT 
            distinguish between other PNG images that may be in the directory; everything
            will be loaded.
        """
        # Get all files matching the WSI file name and correct file type.
        patch_files = np.array(
            [file for file in listdir(self.save_dir) 
            if isfile(join(self.save_dir, file)) and '.png' in file and wsi_name in file])

        patches, targets, coords = [], [], [], []
        for f in patch_files:
            patches.append(np.array(Image.open(self.save_dir + f), dtype=np.uint8))
            targets.append(np.array(Image.open(self.target_dir + f), dtype=np.uint8))
            f_ = f.split('_')
            coords.append([int(f_[1]), int(f_[2])])

        if verbose:
            print("[subslide] loaded", len(patches), "patches from", wsi_name)

        return patches,targets, coords

    ###########################################################################
    #                HDF5-specific helper functions                           #
    ###########################################################################  
    def __get_patches_from_hdf5(self, file_name, verbose=False):
        """ Loads the numpy patches from HDF5 files.
        """
        patches, targets, coords = [], [], [], []
        # Now load the images from H5 file.
        file = h5py.File(self.save_dir + file_name + ".h5",'r+')
        dataset = file['/' + 't']
        new_patches = np.array(dataset).astype('uint8')
        for patch in new_patches:
            patches.append(patch)
        file.close()

        file_target = h5py.File(self.target_dir + file_name + ".h5",'r+')
        dataset = file['/' + 't']
        new_patches = np.array(dataset).astype('uint8')
        for patch in new_patches:
            targets.append(patch)
        file_target.close()

        # Load the corresponding meta.
        with open(self.save_dir + file_name + ".csv", newline='') as metafile:
            reader = csv.reader(metafile, delimiter=' ', quotechar='|')
            for row in reader:
                coords.append([int(row[0]), int(row[1])])
        
        if verbose:
            print("[subslide] loaded from", file_name, ".h5 file", np.shape(patches))

        return patches, targets, coords
        

    
    


