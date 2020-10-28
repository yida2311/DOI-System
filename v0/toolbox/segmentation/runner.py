import torch
import math
import time 
import numpy as np 
from collections import OrderedDict
#from segmentation.dataset.dataset import collate
import sys
sys.path.append("D:/Academic/Django/projects/DOI/v0/segmentation/")
#from segmentation.dataset.dataset import collate
from .dataset.dataset import collate
from .utils.metrics import ConfusionMatrixSeg
from .utils.data import class_to_RGB

def Parallel2Single(original_state):
    """Transform parallel model name to oridinary model"""
    converted = OrderedDict()
    for k, v in original_state.items():
        name = k[7:]
        converted[name] = v
    return converted

def create_model_load_weights(model, ckpt_path=None):
    """Load model weight for model"""
    if not ckpt_path:
        raise ValueError("Don't have checkpoint path")
    state_dict = torch.load(ckpt_path)
    if 'module' in next(iter(state_dict)): # parallel model parameter
        state_dict = Parallel2Single(state_dict)
    state = model.state_dict()
    state.update(state_dict)
    model.load_state_dict(state)
    
    return model


class SlideInference(object):
    def __init__(self, n_class, num_workers, batch_size):
        # self.metrics = ConfusionMatrixSeg(n_class)
        self.n_class = n_class
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def inference(self, dataset, model):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate, shuffle=False, pin_memory=True)
        output = np.zeros((self.n_class, dataset.slide_size[0], dataset.slide_size[1])) # n_class x H x W
        template = np.zeros(dataset.slide_size, dtype='uint8') # H x W
        step = dataset.slide_step

        for sample in dataloader:
            imgs = sample['image']
            coord = sample['coord']
            with torch.no_grad():
                imgs = imgs.cuda()
                preds = model.forward(imgs)
                preds_np = preds.cpu().detach().numpy()
            _, _, h, w = preds_np.shape

            for i in range(imgs.shape[0]):
                x = math.floor(coord[i][0] * step[0])
                y = math.floor(coord[i][1] * step[1])
                output[:, x:x+h, y:y+w] += preds_np[i]
                template[x:x+h, y:y+w] += np.ones((h, w), dtype='uint8')
    
        template[template==0] = 1
        output = output / template
        prediction = np.argmax(output, axis=0)
        
        return class_to_RGB(prediction)


def struct_time():
    # 格式化成2020-08-07 16:56:32
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return cur_time