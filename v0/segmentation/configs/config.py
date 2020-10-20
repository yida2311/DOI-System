config_django = {
    'n_class': 4,
    'img_path': "IMAGES/patch/",
    # 'mask_path': "v0/segmentation/OSCC-Tile/5x_1600/val_masl_1600/",
    'meta_path' :"v0/segmentation/OSCC-Tile/5x_1600/tile_info_val_1600.json",
    'log_path' :"v0/segmentation/results/logs/",
    'output_path' :"IMAGES/Masks/",
    'ckpt_path' :"v0/segmentation/unet-dan-ce-cos120-8.16-52-0.87470.pth",
    'batch_size': 2,
    'num_workers': 2,
}

config = {
    'n_class': 4,
    'img_path': "IMAGES/patch/",
    # 'mask_path': "segmentation/OSCC-Tile/5x_1600/val_masl_1600/",
    # 'meta_path' :"segmentation/OSCC-Tile/5x_1600/tile_info_val_1600.json",
    'log_path' :"segmentation/results/logs/",
    'output_path' :"../IMAGES/Masks/",
    'ckpt_path' :"segmentation/unet-dan-ce-cos120-8.16-52-0.87470.pth",
    'batch_size': 2,
    'num_workers': 2,
}
