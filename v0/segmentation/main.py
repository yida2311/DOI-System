import os 
import time
import torch 
import cv2 
import numpy as np 

from network.seg_generator import generate_unet
from dataset.dataset import OralSlideSeg
from dataset.transform import TransformerSeg
from utils.parser import Options
from runner import create_model_load_weights, SlideInference, struct_time
from utils.metrics import AverageMeter

import sys
sys.path.append('../')

def main(slide_list):
    # arguments
    args = Options().parse()
    n_class = args.n_class
    img_path = args.img_path
    mask_path = args.mask_path
    meta_path = args.meta_path
    log_path = args.log_path
    output_path = args.output_path
    ckpt_path = args.ckpt_path

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
    main(slide_list)





