import os
import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Patch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=4, help='segmentation classes')
        parser.add_argument('--img_path', type=str, help='path to dataset where images store')
        parser.add_argument('--mask_path', type=str, help='path to dataset where masks store')
        parser.add_argument('--meta_path', type=str, help='path to meta_file where images name store')
        parser.add_argument('--output_path', type=str, help='path to store output files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        # parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1: train global branch only. 2: train local branch with fixed global branch. 3: train global branch with fixed local branch')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for image pair')
        # parser.add_argument('--sub_batchsize', type=int, default=2, help=' sub batch size for origin local image (without downsampling)')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers for dataloader')
        parser.add_argument('--ckpt_path', type=str, default="", help='name for seg model path')
        parser.add_argument('--local_rank', type=int, default=0)
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

