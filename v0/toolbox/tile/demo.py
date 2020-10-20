from tile import Cutter

slide_list = ['20190412130806']
file_dir = 'scaledown8_png/'
save_patch_dir = 'tile/patch/'

save_name = 'tile_info_demo.json'
sample_type = 'seg'
patch_size = 1600
overlap = 400
filter_rate = 0.10

cutter = Cutter(slide_list,
                file_dir,
                save_patch_dir,
                sample_type=sample_type)

                
cutter.sample_and_store_patches_png(patch_size,
                                   overlap,
                                   filter_rate,
                                   save_name=save_name)