from torchvision.models.inception import inception_v3
import torch
import torchsummary

# Pretrained inception model
print(f"Loading pretrained inception v3 model...")
inception = inception_v3(pretrained=True).to('cuda')
# inception = torch.nn.Sequential(*(list(inception.children())[:])).to('cuda')

torchsummary.summary(inception, (3, 299, 299))