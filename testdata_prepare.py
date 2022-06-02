import os
from glob import glob
import numpy as np
from PIL import Image

imglist = glob('/home/jovyan/teethimage/final_test/test_withoutB/*.jp*g')

for img_path in imglist:
    fname = os.path.split(img_path)[-1]
    img = np.array(Image.open(img_path))
    pad_B = np.zeros_like(img)
    result = Image.fromarray(np.concatenate([img, pad_B], axis=1))
    result.save('/home/jovyan/teethimage/final_test/test/'+fname)

# results = glob('/home/jovyan/teethimage/final_test/test_results/*.png')
# result_arr = []
# for img_path in results:
#     fname = os.path.split(img_path)[-1]
#     img = np.array(Image.open(img_path))
#     result_arr.append(img)
# result = Image.fromarray(np.concatenate(result_arr, axis=0))
# result.save('/home/jovyan/teethimage/final_test/test_results/test_result.png')