"""Implements the core diffusion algorithms."""
import pickle
import collections

from cleanfid import fid
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import torchvision
from torchvision.utils import save_image
from torchmetrics.image.inception import InceptionScore
import PIL.Image as Image


class Trainer():
    def __init__(
        self, diffusion_model, lr=1e-3, optimizer='adam', 
        folder='.', n_samples=36, from_checkpoint=None,
        weighted_time_sample=False, skip_epochs=0, metrics=None):
        self.model = diffusion_model
        if optimizer=='adam':
            optimizer = Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(optimizer)
        self.optimizer = optimizer
        self.folder = folder
        self.n_samples = n_samples
        self.weighted_time_sample = weighted_time_sample
        if self.weighted_time_sample:
            print('Using weighted time samples.')
            self.time_weights = 1 / (
                0.1  + self.model.sqrt_one_minus_bar_alphas ** 2)
            self.loss_weights = self.time_weights.sum()
        else:
            self.loss_weights = 1.0
            print('Using uniform time sampling.')
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

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                if self.weighted_time_sample:
                    t = torch.multinomial(
                        self.time_weights, batch_size,
                        replacement=True).long().to(self.model.device)
                else:
                    t = torch.randint(
                        0, self.model.timesteps, (batch_size,),
                        device=self.model.device).long()
                loss, metrics = self.model.loss_at_step_t(
                    x0=batch,
                    t=t,
                    loss_weights=self.loss_weights,
                    loss_type='l1')

                if step % 100 == 0:
                    print_line = f'{epoch}:{step}: Loss: {loss.item():.4f}'
                    for key, value in metrics.items():
                        print_line += f' {key}:{value:.4f}'
                        metrics_per_epoch[key].append(value)
                    print(print_line)
                loss.backward()
                self.optimizer.step()
            # save generated images
            self.save_images(epoch, step)
            if epoch % 10 == 9:
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