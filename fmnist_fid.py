from cleanfid import fid
import os
import sys
import data
from PIL import Image
import numpy as np


class FMnist:
    dataset = 'fmnist'
    batch_size = 1
    input_channels = 1
    input_size = 32
    data_dir = './root'
    images_path = 'root/FashionMNIST/images'

if __name__ == '__main__':
    config = FMnist()
    if not os.path.exists(config.images_path):
        os.makedirs(config.images_path)
    fid.remove_custom_stats('fmnist', mode='clean')
    for counter, (batch, _) in enumerate(data.get_dataset(config)):
        image = batch[0][0].detach().cpu().numpy()
        image = (255 * (0.5 * (1 + image))).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f'{config.images_path}/{counter}.png')
    fid.make_custom_stats('fmnist', config.images_path, mode='clean')
    fid_mean = fid.compute_fid(
        config.images_path,
        dataset_name='fmnist',
        dataset_res=config.input_size,
        batch_size=100,
        dataset_split='custom',
        z_dim=config.input_channels * config.input_size ** 2)
    print('FID score of the dataset:', fid_mean)