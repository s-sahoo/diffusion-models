import argparse
import os

import numpy as np
import torch

import data
from models.unet.standard import UNet
from models.unet.colab import Unet as colab_UNet
from models.unet.biheaded import BiheadedUNet
from models.modules import feedforward
from models.unet.auxiliary import AuxiliaryUNet, TimeEmbeddingAuxiliaryUNet
# from data import get_data_loader
from diffusion.gaussian import GaussianDiffusion
from diffusion.auxiliary import InfoMaxDiffusion
from diffusion.learned import LearnedGaussianDiffusion
from diffusion.learned_blurring import Blurring
from diffusion.learned_masking import Masking
from models.modules.encoders import ConvGaussianEncoder
# from data.fashion_mnist import FashionMNISTConfig
from trainer.gaussian import Trainer, process_images
from misc.eval.sample import sample, viz_latents

import metrics
# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands')

    # train

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'blur', 'masking', 'learned'], 
        help='type of ddpm model to run')
    train_parser.add_argument('--schedule', default='cosine',
        choices=['linear', 'cosine'], 
        help='constants scheduler for the diffusion model.')
    train_parser.add_argument('--sampler', default='naive',
        choices=['naive', 'momentum'], 
        help='Sampler type during the inference phase.')
    train_parser.add_argument('--loss_type', default='elbo',
        choices=['elbo', 'soft_diffusion'], 
        help='loss functions used in diffusion models.')
    train_parser.add_argument('--timesteps', type=int, default=200,
        help='total number of timesteps in the diffusion model')
    train_parser.add_argument('--weighted_time_sample', type=bool, default=False,
        help='total number of timesteps in the diffusion model')
    train_parser.add_argument('--dataset', default='fmnist',
        choices=['fmnist', 'mnist'], help='training dataset')
    train_parser.add_argument('--checkpoint', default=None,
        help='path to training checkpoint')
    train_parser.add_argument('-e', '--epochs', type=int, default=50,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=128,
        help='training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.0005,
        help='learning rate')
    train_parser.add_argument('--optimizer', default='adam', choices=['adam'],
        help='optimization algorithm')
    train_parser.add_argument('--folder', default='.',
        help='folder where logs will be stored')
    # masking parameters
    train_parser.add_argument('--fixed_masking', type=bool, default=False,
        help='Fixed Masking.')
    train_parser.add_argument('--margin', type=float, default=0.0,
        help='margin for the weights.')
    # blur parameters
    train_parser.add_argument('--transform_type', default='blur',
        choices=['blur', 'learnable_forward'], 
        help='constants scheduler for the diffusion model.')
    train_parser.add_argument('--level_initializer', default='random',
        choices=['linear', 'zero', 'random'], 
        help='constants scheduler for the diffusion model.')
    train_parser.add_argument('--drop_forward_coef', type=bool, default=False,
        help='Dont scale image in the forward pass')
    train_parser.add_argument('--levels_no_reparam', type=bool, default=False,
        help='Dont further reparameterize blur variables.')
    train_parser.add_argument('--fixed_blur', type=bool, default=False,
        help='total number of timesteps in the diffusion model')
    
    # eval

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(func=eval)

    eval_parser.add_argument('--schedule', default='cosine',
        choices=['linear', 'cosine'], 
        help='constants scheduler for the diffusion model.')
    eval_parser.add_argument('--model', default='gaussian',
        choices=['gaussian', 'infomax', 'blur', 'learned', 'learned_input_time'], 
        help='type of ddpm model to run')
    eval_parser.add_argument('--dataset', default='fmnist',
        choices=['fmnist', 'mnist'], help='training dataset')
    eval_parser.add_argument('--timesteps', type=int, default=200,
        help='total number of timesteps in the diffusion model')
    eval_parser.add_argument('--batch-size', type=int, default=128,
        help='training batch size')
    eval_parser.add_argument('--checkpoint', required=True,
        help='path to training checkpoint')
    eval_parser.add_argument('--deterministic', action='store_true', 
        default=False, help='run in deterministic mode')
    eval_parser.add_argument('--sample', type=int, default=None,
        help='how many samples to draw')
    eval_parser.add_argument('--interpolate', type=int, default=None,
        help='how many samples to interpolate')
    eval_parser.add_argument('--latents', type=int, default=None,
        help='how many points to visualize in latent space')
    eval_parser.add_argument('--folder', default='.',
        help='folder where output will be stored')
    eval_parser.add_argument('--name', default='test-run',
        help='name of the files that will be saved')
    eval_parser.add_argument('--sampler', default='naive',
        choices=['naive', 'momentum'], 
        help='Sampler type during the inference phase.')
    eval_parser.add_argument('--loss_type', default='elbo',
        choices=['elbo', 'soft_diffusion'], 
        help='loss functions used in diffusion models.')
    # masking parameters
    eval_parser.add_argument('--fixed_masking', type=bool, default=False,
        help='Fixed Masking.')
    eval_parser.add_argument('--margin', type=float, default=0.0,
        help='margin for the weights.')
    # blur parameters
    eval_parser.add_argument('--transform_type', default='blur',
        choices=['blur', 'learnable_forward'], 
        help='constants scheduler for the diffusion model.')
    eval_parser.add_argument('--level_initializer', default='random',
        choices=['linear', 'zero', 'random'], 
        help='constants scheduler for the diffusion model.')
    eval_parser.add_argument('--drop_forward_coef', type=bool, default=False,
        help='Dont scale image in the forward pass')
    eval_parser.add_argument('--levels_no_reparam', type=bool, default=False,
        help='Dont further reparameterize blur variables.')
    eval_parser.add_argument('--fixed_blur', type=bool, default=False,
        help='total number of timesteps in the diffusion model')
    return parser

# ----------------------------------------------------------------------------

def train(args):
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data.get_dataset_config(args)
    model = get_model(args, device)

    if args.checkpoint:
        model.load(args.checkpoint)
    trainer = Trainer(
        model,
        weighted_time_sample=args.weighted_time_sample,
        lr=args.learning_rate,
        optimizer=args.optimizer,
        folder=args.folder,
        from_checkpoint=args.checkpoint,
    )
    trainer.fit(data.get_dataset(args), args.epochs)


def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data.get_dataset_config(args)
    model = get_model(args, device)
    model.load(args.checkpoint)
    model.reverse_model.eval()

    if args.sample:
        path = f'{args.folder}/{args.name}-samples.png'
        sample(model, args.sample, path, args.deterministic)

    scores = {'fid_score': [], 'is_score': []}
    fid_score = metrics.FID()
    inception_score = metrics.InceptionMetric()
    for batch, _ in data.get_dataset(args):
        real_images = process_images(
            batch.to(model.device).detach().cpu().numpy())
        samples = model.sample(real_images.shape[0])[-1]
        samples = process_images(samples)
        fid_mean = fid_score.calculate_frechet_distance(
            real_images, samples)
        scores['fid_score'].append(fid_mean)
        break
        # is_mean, _ = inception_score.compute_inception_scores(
        #     (255 * samples).type(torch.uint8))
        # scores['is_score'].append(is_mean)
    print('FID score: {:.2f}'.format(np.mean(scores['fid_score'])))
    print('IS score: {:.2f}'.format(np.mean(scores['is_score'])))

# ----------------------------------------------------------------------------

def get_config(args):
    if args.dataset == 'fmnist':
        return FashionMNISTConfig
    else:
        raise ValueError()

def get_model(config, device):
    if args.model == 'gaussian':
        model = create_gaussian(config, device)
    elif args.model == 'infomax':
        model = create_infomax(config, device)
    elif args.model == 'blur':
        model = create_blur(config, device)
    elif args.model == 'learned':
        model = create_learned(config, device)
    elif args.model == 'learned_input_time':
        model = create_learned_input_time(config, device, args.reparam)
    elif args.model == 'masking':
        model = create_masked(config, device)
    else:
        raise ValueError(args.model)
    return model

def create_gaussian(config, device):
    img_shape = [config.input_channels, config.input_size, config.input_size]
    model = UNet(
        channels=config.unet_channels,
        img_shape=img_shape,
    ).to(device)

    return GaussianDiffusion(
        model=model,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

def create_infomax(config, device):
    img_shape = [config.input_channels, config.input_size, config.input_size]
    a_shape = [config.a_dim, 1, 1]

    a_encoder = ConvGaussianEncoder(
        img_shape=img_shape,
        a_shape=a_shape,
    ).to(device)

    # model = AuxiliaryUNet(
    model = TimeEmbeddingAuxiliaryUNet(
        channels=config.unet_channels,
        img_shape=img_shape,
        a_shape=a_shape,
    ).to(device)

    return InfoMaxDiffusion(
        noise_model=model,
        a_encoder_model=a_encoder,
        timesteps=config.timesteps,
        img_shape=img_shape,
        a_shape=a_shape,
        device=device,
    )

def create_learned(config, device):
    img_shape = [config.input_channels, config.input_size, config.input_size]
    z_shape = img_shape.copy()

    z_encoder = ConvGaussianEncoder(
        img_shape=img_shape,
        a_shape=z_shape,
    ).to(device)

    model = UNet(
        channels=config.unet_channels,
        img_shape=img_shape,
    ).to(device)

    return LearnedGaussianDiffusion(
        noise_model=model,
        z_encoder_model=z_encoder,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
    )

def create_masked(config, device):
    img_shape = (config.input_channels,
                 config.input_size,
                 config.input_size)

    reverse_model = UNet(
        channels=config.unet_channels,
        img_shape=img_shape,
    ).to(device)

    return Masking(
        noise_model=None,
        forward_matrix=None,
        reverse_model=reverse_model,
        fixed_masking=args.fixed_masking,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
        loss_type=args.loss_type,
        margin=args.margin,
    )


def create_blur(config, device):
    img_shape = (config.input_channels,
                 config.input_size,
                 config.input_size)
    
    reverse_model = UNet(
        channels=config.unet_channels,
        img_shape=img_shape,
    ).to(device)

    return Blurring(
        noise_model=None,
        forward_matrix=None,
        reverse_model=reverse_model,
        fixed_blur=args.fixed_blur,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=config.timesteps,
        device=device,
        drop_forward_coef=args.drop_forward_coef,
        levels_no_reparam=args.levels_no_reparam,
        level_initializer=args.level_initializer,
        loss_type=args.loss_type,
        transform_type=args.transform_type,
        sampler=args.sampler,
    )


# ----------------------------------------------------------------------------

if __name__ == '__main__':
  parser = make_parser()
  args = parser.parse_args()
  args.func(args)