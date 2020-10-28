import os 
import time
import torch 
import cv2 
import numpy as np 

# from segmentation.network.seg_generator import generate_unet
# from segmentation.dataset.dataset import OralSlideSeg
# from segmentation.dataset.transform import TransformerSeg
# from segmentation.runner import create_model_load_weights, SlideInference, struct_time
# from segmentation.utils.metrics import AverageMeter

from .network.seg_generator import generate_unet
from .dataset.dataset import OralSlideSeg
from .dataset.transform import TransformerSeg
from .runner import create_model_load_weights, SlideInference, struct_time
from .utils.metrics import AverageMeter

import sys
sys.path.append('../')

class Segmentation():
    def __init__(self, config, slide_list, info):
        self.n_class = config['n_class']
        self.img_path = config['img_path']
        #self.mask_path = config['mask_path']
        # self.meta_path = config['meta_path']
        self.log_path = config['log_path']
        self.output_path = config['output_path']
        self.ckpt_path = config['ckpt_path']
        self.slide_list = slide_list
        self.info = info # {"slide": {"size":[h, w], "tiles": [x, y], "step":[step_x, step_y]}}

        if not os.path.exists(self.log_path): 
            os.makedirs(self.log_path)
        if not os.path.exists(self.output_path): 
            os.makedirs(self.output_path)

        self.f_log = open(self.log_path + "test.log", 'a+')
        self.f_log.write("%s  start segmentation \n"%(struct_time()))
        ###################################
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.slide_time = AverageMeter("DataTime", ':3.3f')

        self.transformer = TransformerSeg
        self.dataset = OralSlideSeg(self.slide_list, self.img_path, self.info, mask_dir=None, label=False, transform=self.transformer)
        self.num_slides = len(self.dataset.slides)

        ###################################
        self.f_log.write("%s  load models...... \n"%(struct_time()))
        self.model = generate_unet(num_classes=self.n_class, encoder_name='resnet34')
        self.model = create_model_load_weights(self.model, ckpt_path=self.ckpt_path)
        self.model.cuda()

        #######################################
        self.evaluator = SlideInference(self.n_class, self.num_workers, self.batch_size) 
        self.f_log.write("%s   ***%d slides*** \n"%(struct_time(), self.num_slides))
        self.f_log.close()  

    def run_segmentation(self, slide_name):
        self.f_log = open(self.log_path + "test.log", 'a+')

        #dataset = OralSlideSeg([slide_name], self.img_path, self.meta_path, mask_dir=self.mask_path, label=False, transform=self.transformer)
        #num_slides = len(dataset.slides)

        find_list = [1 if ele==slide_name else 0 for ele in self.slide_list]
        find_index = [i for i in range(self.num_slides) if find_list[i]==1]
        find_index = find_index[0]
        start_time = time.time()
        self.dataset.get_patches_from_index(find_index)
        output = self.evaluator.inference(self.dataset, self.model)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.output_path, self.dataset.slide+'.png'), output)
        self.slide_time.update(time.time()-start_time)
        self.f_log.write("%s  [%d] %s \n"%(struct_time(), find_index+1, self.dataset.slide))
        self.f_log.flush()    

        self.f_log.write("[Time consuming: %.3f][%.3fs per slide]\n"%(self.slide_time.sum, self.slide_time.avg))
        self.f_log.close()    
    #return slide_name

def segmentation_api(slide_list):
    # arguments

    n_class = 4
    img_path = "../OSCC-Tile/5x_1600/val_1600/"
    mask_path = "../OSCC-Tile/5x_1600/val_masl_1600/"
    meta_path = "../OSCC-Tile/5x_1600/tile_info_val_1600.json"
    log_path = "results/logs/"
    output_path = "results/predictions/"
    ckpt_path = "./unet-resnet34-dan-1e-4-adam-cos-sce-7.15-74-0.8582.pth"

    if not os.path.exists(log_path): 
        os.makedirs(log_path)
    if not os.path.exists(output_path): 
        os.makedirs(output_path)

    f_log = open(log_path + "test.log", 'w')
    f_log.write("%s  start segmentation \n"%(struct_time()))
    ###################################
    batch_size = args.batch_size
    num_workers = args.num_workers
    slide_time = AverageMeter("DataTime", ':3.3f')

    transformer = TransformerSeg
    dataset = OralSlideSeg(slide_list, img_path, meta_path, mask_dir=mask_path, label=False, transform=transformer)

    ###################################
    f_log.write("%s  load models...... \n"%(struct_time()))
    model = generate_unet(num_classes=n_class, encoder_name='resnet34')
    model = create_model_load_weights(model, ckpt_path=ckpt_path)
    model.cuda()

    #######################################
    evaluator = SlideInference(n_class, num_workers, batch_size)
    num_slides = len(dataset.slides)
    f_log.write("%s   ***%d slides*** \n"%(struct_time(), num_slides))
    for i in range(num_slides):
        start_time = time.time()
        dataset.get_patches_from_index(i)
        output = evaluator.inference(dataset, model)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_path, dataset.slide+'_ouput.png'), output)
        slide_time.update(time.time()-start_time)
        f_log.write("%s  [%d] %s \n"%(struct_time(), i+1, dataset.slide))
        f_log.flush()

    f_log.write("[Time consuming: %.3f][%.3fs per slide]\n"%(slide_time.sum, slide_time.avg))
    f_log.close()

if __name__ == '__main__':
    slide_list = ['_20190412130806', '_20190718213917']
    #slide_list = ['_20190412130806']
    segmentation_api(slide_list)





