import argparse
import os
import pickle

from cleanfid import fid
import numpy as np
import torch
import PIL.Image as Image

import data
from models.unet.standard import UNet
from models.unet.colab import Unet as colab_UNet
from models.unet.biheaded import BiheadedUNet
from models.modules import feedforward
from models.unet.auxiliary import AuxiliaryUNet, TimeEmbeddingAuxiliaryUNet
from diffusion.gaussian import GaussianDiffusion
from diffusion.auxiliary import InfoMaxDiffusion
from diffusion.learned import LearnedGaussianDiffusion
from diffusion.learned_blurring import Blurring
from diffusion.learned_masking import Masking
from models.modules.encoders import ConvGaussianEncoder
from trainer.gaussian import Trainer
from misc.eval.sample import sample, viz_latents

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

    eval_parser.add_argument('--folder', default='.',
        help='folder to evaluate.')
    eval_parser.add_argument('--sampler', default='naive',
        choices=['naive', 'momentum'], 
        help='Sampler type during the inference phase.')
    eval_parser.add_argument('--model_selection', default='fid_score',
        choices=['fid_score', 'last', 'total_loss'], 
        help='pick the best model.')
    return parser

# ----------------------------------------------------------------------------

def find_recent_checkpoint(folder):
    max_checkpoint_epoch = -1
    checkpoint = None
    for i in os.listdir(folder):
        if 'pth' in i:
            max_checkpoint_epoch = max(
                max_checkpoint_epoch,
                int(i.split('-')[-1][:-4]))
            checkpoint = os.path.join(
                folder,
                f'model-{max_checkpoint_epoch}.pth')
    return checkpoint, max_checkpoint_epoch + 1


def find_checkpoint(folder, metric):
    if metric == 'last':
        checkpoint, _ = find_recent_checkpoint(folder)
        return checkpoint
    best_score = np.float('inf')
    best_epoch = None
    with open(f'{folder}/{metric}.txt', 'r') as f:
        for line in f.readlines():
            epoch, score = line.strip().split()
            if float(score) < best_score:
                best_score = float(score)
                best_epoch = int(epoch)
    return os.path.join(folder, f'model-{best_epoch}.pth')


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    data.get_dataset_config(args)
    model = get_model(args, device)
    with open(f'{args.folder}/args.pkl', 'wb') as f:
        args.func = None
        pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Checkpoints
    checkpoint = args.checkpoint
    skip_epochs = 0
    metrics = None
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    else:
        checkpoint, skip_epochs = find_recent_checkpoint(args.folder)
        if checkpoint is None:
            print(f'No checkpoints found in {args.folder}')
        metrics_file = f'{args.folder}/metrics.pkl'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                metrics = pickle.load(f)

    trainer = Trainer(
        model,
        weighted_time_sample=args.weighted_time_sample,
        lr=args.learning_rate,
        optimizer=args.optimizer,
        folder=args.folder,
        from_checkpoint=checkpoint,
        skip_epochs=skip_epochs,
        metrics=metrics,
    )
    trainer.fit(data.get_dataset(args), args.epochs)


def eval(cmd_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(f'{cmd_args.folder}/args.pkl', 'rb') as f:
        args = pickle.load(f)
    args.sampler = cmd_args.sampler
    data.get_dataset_config(args)

    model = get_model(args, device)
    checkpoint = find_checkpoint(cmd_args.folder, cmd_args.model_selection)
    model.load(checkpoint)
    model.reverse_model.eval()
    
    eval_name = f'{cmd_args.sampler}-{cmd_args.model_selection}'
    sample(model,
           36,
           '{}/eval-{}-samples.png'.format(args.folder, eval_name),
           False)

    scores = {'fid_score': [], 'is_score': []}
    fid_score = model.compute_fid_scores(
        batch_size=args.batch_size,
        num_samples=10000)
    with open('{}/eval-{}-fid.txt'.format(args.folder, eval_name), 'w') as f:
        f.write(str(fid_score))
    print('FID score: {:.2f}'.format(fid_score))

# ----------------------------------------------------------------------------

def get_model(config, device):
    if config.model == 'gaussian':
        model = create_gaussian(config, device)
    elif config.model == 'infomax':
        model = create_infomax(config, device)
    elif config.model == 'blur':
        model = create_blur(config, device)
    elif config.model == 'learned':
        model = create_learned(config, device)
    elif config.model == 'learned_input_time':
        model = create_learned_input_time(config, device, args.reparam)
    elif config.model == 'masking':
        model = create_masked(config, device)
    else:
        raise ValueError(config.model)
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


def create_blur(args, device):
    img_shape = (args.input_channels,
                 args.input_size,
                 args.input_size)
    
    reverse_model = UNet(
        channels=args.unet_channels,
        img_shape=img_shape,
    ).to(device)

    return Blurring(
        noise_model=None,
        forward_matrix=None,
        reverse_model=reverse_model,
        fixed_blur=args.fixed_blur,
        schedule=args.schedule,
        img_shape=img_shape,
        timesteps=args.timesteps,
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
