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
import torchvision
from torchvision.models.inception import inception_v3
from scipy import linalg
import os
import numpy as np

 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# Pretrained inception model
print(f"Loading pretrained inception v3 model...")

def calculate_frechet_distance(mu1, mu2, cov1, cov2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(cov1)
    sigma2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)+ np.trace(sigma2) - 2 * tr_covmean)

class PartialInceptionNetwork(torch.nn.Module):
    def __init__(self, transform_input=False):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations
inception = PartialInceptionNetwork()


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
    opt.epoch = 90
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.eval()
    inception.to(model.device)
    inception.eval()
    resize_layer = torchvision.transforms.Resize((299, 299)).to(model.device)
    print(model)

    pred_features, true_features = [], []
    predictions, gts, inputs = [], [], []
    print("Calculating FID...")

    for i, test_data in tqdm(enumerate(dataset), total=len(dataset)):
        model.set_input(test_data)
        model.test()
        # print(model.fake_B.shape,'\n', model.real_B.shape)
        
        predictions.append(torch.permute(torch.squeeze(model.fake_B), (1, 2, 0)).detach().cpu().numpy())
        gts.append(torch.permute(torch.squeeze(model.real_B), (1, 2, 0)).detach().cpu().numpy())
        inputs.append(torch.permute(torch.squeeze(model.real_A), (1, 2, 0)).detach().cpu().numpy())

        # feature extraction
        pred_feature = inception(resize_layer(model.fake_B))
        true_feature = inception(resize_layer(model.real_B))
        pred_features.append(pred_feature.detach().cpu().numpy())
        true_features.append(true_feature.detach().cpu().numpy())

    path = model.save_dir+f'/{opt.phase}_new_results'
    createFolder(path)
    for i, result in tqdm(enumerate(list(zip(inputs, predictions, gts))), total=len(inputs)):
        out = (np.concatenate(result, axis=1)*0.5 + 0.5) * 255
        # print(out.shape)
        out = Image.fromarray(out.astype(np.uint8))
        out.save(path+f'/{i+1}.png')
    # estimate the distribution - assuming the gaussian distribution
    pred_features, true_features = np.concatenate(pred_features), np.concatenate(true_features)
    mu_pred, cov_pred = np.mean(pred_features, axis=0), np.cov(pred_features)
    mu_true, cov_true = np.mean(true_features, axis=0), np.cov(true_features)
    # if the number of data exceeds 2,048, eps = 0
    eps = 0 if opt.phase == 'test' or opt.phase == 'train' else 1e-7
    fid = calculate_frechet_distance(mu_pred, mu_true, cov_pred, cov_true, eps)
    print(f"\t {opt.phase} FID: {fid:5f}")
    with open(os.path.join(model.save_dir, f"{opt.phase}_fid.txt"), 'a') as f:
        f.write(f"{opt.name} {opt.phase} FID: {fid}\n")
