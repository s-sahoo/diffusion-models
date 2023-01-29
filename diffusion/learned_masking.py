"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .gaussian import GaussianDiffusion


class TopK(torch.autograd.Function):
    """
    Computes topk entries.
    """

    @staticmethod
    def forward(ctx, weights, threshold_indices, img_dim, device):
        """
        :param ctx: context for backpropagation
        :param weights (torch.Tensor of shape [img_dim, img_dim]): vertex weights
        :return: shortest paths (torch.Tensor of shape [batch_size, img_dim, img_dim]): indicator matrices
        of taken paths
        """
        thresholds, _ = torch.sort(weights.view(-1))

        thresholds = get_by_idx(
            thresholds.cpu(),
            threshold_indices,
            thresholds.shape)
        thresholds = torch.ones(
            (1, img_dim, img_dim), device=device,
        ) * thresholds[:, None, None]
        return (weights > thresholds).type(weights.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # In the forward pass weights with shape (img_dim, img_dim)
        # is broadcasted to (batch_size, img_dim, img_dim).
        # Hence, in the backward pass to take mean to kill the extra dim.
        return grad_output.mean(dim=0), None, None, None


class Masking(GaussianDiffusion):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, forward_matrix, reverse_model, timesteps,
        img_shape, schedule='cosine', fixed_masking=False, device='cpu',
        loss_type='elbo', margin=0.0):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device)
        self.loss_type = loss_type
        self.drop_forward_coef = True
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape
        self.reverse_model = reverse_model
        self.fixed_masking = fixed_masking
        self.margin = margin
        print(margin, self.margin)
        self.mask_params = torch.nn.Parameter(
            torch.rand((self.img_dim, self.img_dim), device=self.device))
        if fixed_masking:
            print('Fixed Masking.')
        else:
            print('Learnable Masking.')

        self.identity = torch.eye(
            self.img_dim,
            dtype=torch.float32,
            device=self.device)[None, :, :]

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None:
            x = torch.randn(shape, device=self.device)
        xs = []

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs


    def _forward_sample(self, x0, time):
        weights = self.mask_params
        if self.fixed_masking:
            weights = weights.detach()
        weights = weights + self.margin * torch.bernoulli(
            0.5 * torch.ones_like(weights))
        weights = weights - weights.mean()
        weights = weights / torch.norm(weights)
        threshold_indices = (time * (self.img_dim ** 2 - 1) / (
            self.timesteps - 1)).type(torch.int64)
        return TopK.apply(
            weights, threshold_indices, self.img_dim, self.device)


    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transformation_matrices = self._forward_sample(x0, t)
        x0 = x0.view(batch_size, self.img_dim, self.img_dim)
        masked_images = (transformation_matrices * x0).view(
            * original_batch_shape)
        x_t = self._add_noise(masked_images, t, noise)
        return x_t

    # def _momentum_sampler(self, xt, t):
    #     x0_approx = xt + self.reverse_model(xt, t)
    #     yt_hat = self.q_sample(x0_approx, t, torch.zeros_like(x0_approx))
    #     yt_hat = self.q_sample(x0_approx, t, torch.zeros_like(x0_approx))
    #     eta = torch.randn_like(x0_approx)
    #     epsilon = yt_hat - xt
    #     bar_alphas = get_by_idx(self.bar_alphas, T, x.shape)
    #     bar_alphas_prev = get_by_idx(
    #         self.bar_alphas_prev, T, x.shape)
    #     sigma = torch.sqrt(1 - bar_alphas)
    #     sigma_prev = torch.sqrt(1 - bar_alphas_prev)
    #     z = x - ((sigma_prev / sigma) ** 2 - 1) * epsilon + (
    #         sigma_t ** 2 - sigma_prev ** 2).sqrt() * eta
    #     return z + self.reverse_model(xt, t - 1) - x0_approx

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
        x0_approx = xt + self.reverse_model(xt, t)

        if t_index == 0:
            return x0_approx
        noise = None
        if deterministic:
            noise = torch.zeros_like(x0_approx)
        return self.q_sample(x0_approx, t - 1, noise)

    def _compute_prior_kl_divergence(self, x0, batch_size):
        transformation_matrices = self._get_blur_matrices(
            x0,
            torch.tensor([self.timesteps - 1] * batch_size,
                device=self.device),
            mask=0)
        # print(transformation_matrices.view(batch_size, -1).sum(dim=-1))
        mu_squared = (
            transformation_matrices * x0.view(
                batch_size, self.img_dim, self.img_dim))
        mu_squared = (mu_squared ** 2).view(batch_size, -1).sum(dim=1).mean()
        return 0.5 * mu_squared

    def _get_blur_matrices(self, x0, t, mask):
        transformation_matrices = self._forward_sample(x0, t)
        return ((1 - mask) * transformation_matrices
                + mask * self.identity)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            raise NotImplementedError()
        batch_size = x0.shape[0]
        x_t = self.q_sample(x0=x0, t=t, noise=noise)

        if self.loss_type == 'elbo':
            transformation_matrices = self._get_blur_matrices(
                x0,
                (t - 1) * (t > 0),
                mask=(t == 0).type(x0.dtype)[:, None, None])
        elif self.loss_type == 'soft_diffusion':
            transformation_matrices = self._get_blur_matrices(
                x0, t, mask=0)

        target = transformation_matrices * (
            x0 - x_t).view(batch_size, self.img_dim, self.img_dim)
        prediction = transformation_matrices * self.reverse_model(
            x_t, t).view(batch_size, self.img_dim, self.img_dim)
        reconstruction_loss = self.p_loss_at_step_t(target, prediction, 'l2')
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)
        total_loss = reconstruction_loss + kl_divergence / self.timesteps

        return total_loss, {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
