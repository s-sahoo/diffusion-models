"""Implements the core diffusion algorithms."""
import pickle
import collections

from cleanfid import fid
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
import PIL.Image as Image


class Trainer():
    def __init__(
        self, diffusion_model, lr=1e-3, optimizer='adam', 
        folder='.', n_samples=36, from_checkpoint=None,
        loss_coefficients=False, skip_epochs=0, metrics=None,
        loss_epsilon=1e-6, loss_type='l2'):
        self.model = diffusion_model
        if optimizer=='adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(optimizer)
        self.optimizer = optimizer
        self.folder = folder
        self.n_samples = n_samples
        self.loss_coefficients = loss_coefficients
        self.loss_epsilon = loss_epsilon
        self.loss_type = loss_type
        print('Using weighted time samples:', self.loss_coefficients)
        if metrics is None:
            self.metrics = collections.defaultdict(list)
        else:
            self.metrics = metrics
        self.skip_epochs = skip_epochs
        if from_checkpoint is not None:
            self.load_model(from_checkpoint)

    def fit(self, data_loader, epochs):
        for epoch in range(self.skip_epochs, epochs, 1):
            metrics_per_epoch = collections.defaultdict(list)
            for step, (batch, _) in enumerate(data_loader):
                self.optimizer.zero_grad()
                batch_size = batch.shape[0]
                batch = batch.to(self.model.device)

                loss_coefficients = 1 / (
                    self.loss_epsilon + self.model._get_noise_sigma().detach() ** 2)
                if self.loss_coefficients == 'sample':
                    t = torch.multinomial(
                        loss_coefficients, batch_size,
                        replacement=True).long().to(self.model.device)
                    loss_weights = loss_coefficients.sum()
                elif self.loss_coefficients in {'ignore', 'scale'}:
                    t = torch.randint(
                        0, self.model.timesteps, (batch_size,),
                        device=self.model.device).long()
                    if self.loss_coefficients == 'scale':
                        loss_scale = loss_coefficients
                    else:
                        loss_scale = torch.ones_like(
                            loss_coefficients, device=self.model.device)
                    loss_weights = 1.0
                loss, metrics = self.model.loss_at_step_t(
                    x0=batch,
                    t=t,
                    loss_weights=loss_weights,
                    loss_scale=loss_scale,
                    loss_type=self.loss_type)

                if step % 100 == 0:
                    print_line = f'{epoch}:{step}: Loss: {loss.item():.4f}'
                    for key, value in metrics.items():
                        print_line += f' {key}:{value:.4f}'
                        metrics_per_epoch[key].append(value)
                    print(print_line)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
            # save generated images
            self.save_images(epoch, step)
            if epoch % 5 == 4:
                self.compute_fid_scores(batch_size, epoch)
            self.record_metrics(epoch, metrics_per_epoch)
            self.save_model(epoch)
            self.write_metrics()

    def write_metrics(self):
        with open(f'{self.folder}/metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        for key, values in self.metrics.items():
            with open(f'{self.folder}/{key}.txt', 'w') as f:
                for value in values:
                    f.write(value)

    def record_metrics(self, epoch, metrics):
        for key, value in metrics.items():
            self.metrics[key].append(f'{epoch} {np.mean(value)}\n')

    def compute_fid_scores(self, batch_size, epoch):
        score = self.model.compute_fid_scores(
            batch_size=batch_size,
            num_samples=10 * batch_size)
        self.metrics['fid_score'].append(f'{epoch} {score}\n')
        print('FID score: {:.2f}'.format(score))

    def save_images(self, epoch, step):
        samples = torch.Tensor(self.model.sample(self.n_samples)[-1])
        samples = (samples + 1) * 0.5
        print(samples.min(), samples.max())
        path = f'{self.folder}/sample-{epoch}-{step}.png'
        save_image(samples, path, nrow=6)

    def save_model(self, epoch):
        path = f'{self.folder}/model-{epoch}.pth'
        self.model.save(path)
        print(f'Saved PyTorch model state to {path}')

    def load_model(self, path):
        self.model.load(path)
        print(f'Loaded PyTorch model state from {path}')