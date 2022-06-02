import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# data_dir = '/home/jovyan/teethimage/pytorch-CycleGAN-and-pix2pix/datasets/teethimgs'
# data_dir = '/home/jovyan/teethimage/final_test'
# model_ver = 'titr_pix2pix_resnet9_312Lo_256Cr'
# model_name = 'pix2pixTeethImg'

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.checkpoints_dir = '/home/jovyan/teethimage/pytorch-CycleGAN-and-pix2pix/checkpoints'
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1

# opt.dataroot = data_dir
opt.crop_size = 256
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt.rotate=False
opt.phase='test'
opt.epoch=90 
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# model.eval()

predictions, inputs = [], []
for i, test_data in tqdm(enumerate(dataset), total=len(dataset)):
    model.set_input(test_data)
    model.test()
    predictions.append(torch.permute(torch.squeeze(model.fake_B), (1, 2, 0)).detach().cpu().numpy())
    inputs.append(torch.permute(torch.squeeze(model.real_A), (1, 2, 0)).detach().cpu().numpy())

path = '/home/jovyan/teethimage/final_test/test_VGG_results'
createFolder(path)
for i, result in tqdm(enumerate(list(zip(inputs, predictions))), total=len(inputs)):
    out = (np.concatenate(result, axis=1)*0.5 + 0.5) * 255
    # print(out.shape)
    out = Image.fromarray(out.astype(np.uint8))
    out.save(path+f'/{i+1}.png')