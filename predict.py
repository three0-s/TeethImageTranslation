# author@ 
#   Yewon Lim(ga06033@yonsei.ac.kr) 
# date@ 
#   2022.04.26
# =====================================
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from glob import glob
 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.rotate=False
    opt.phase='test'
    opt.epoch = 'latest'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    path_pred = model.save_dir+f'/{opt.phase}_results_{opt.epoch}/pred'
    path_gt = model.save_dir+f'/{opt.phase}_results_{opt.epoch}/gt'

    createFolder(path_pred)
    createFolder(path_gt)
    pre_gts = glob(path_gt+'/*.png')
    save_gt = pre_gts != len(dataset)
    for i, test_data in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(test_data)
        model.test()
       
        pred = (torch.permute(torch.squeeze(model.fake_B), (1, 2, 0)).detach().cpu().numpy()*0.5 + 0.5)*255
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save(path_pred+f'/{i+1}.png')
        if save_gt:
            gt = (torch.permute(torch.squeeze(model.real_B), (1, 2, 0)).detach().cpu().numpy()*0.5 + 0.5)*255
            gt = Image.fromarray(gt.astype(np.uint8))
            gt.save(path_gt+f'/{i+1}.png')

    
    
    
    # for i, result in tqdm(enumerate(predictions), total=len(predictions)):
    #     out = (result*0.5 + 0.5) * 255
    #     # print(out.shape)
    #     out = Image.fromarray(out.astype(np.uint8))
    #     out.save(path_pred+f'/{i+1}.png')
    
    # pre_gts = glob(path_gt+'/*.png')
    # if pre_gts == []:
    #     for i, result in tqdm(enumerate(gts), total=len(gts)):
    #         out = (result*0.5 + 0.5) * 255
    #         # print(out.shape)
    #         out = Image.fromarray(out.astype(np.uint8))
    #         out.save(path_gt+f'/{i+1}.png')