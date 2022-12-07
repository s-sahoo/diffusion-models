"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .gaussian import GaussianDiffusion

class LearnedGaussianDiffusion(GaussianDiffusion):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, forward_matrix, timesteps, img_shape,
        alpha=5.0, schedule='linear', device='cpu'
    ):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device
        )
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape

    def _forward_sample(self, x0, time):
        return self.forward_matrix(self.model.time_mlp(time))

    def _get_alpha(self, batch_size):
        time = torch.tensor([self.timesteps] * batch_size,
                             device=self.device)
        m_T = self._forward_sample(
            self, None, time).view(batch_size, -1)
        return 1 / (np.sqrt(self.timesteps) * m_T)

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        # encode x0 into auxiliary encoder variable a
        if noise is None:  noise, _, _ = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transormation_matrices = self._forward_sample(
            x0, t).view(batch_size, -1)
        x0 = x0.view(batch_size, -1)
        t = t.view(batch_size, -1)
        noise = noise.view(batch_size, -1)
        alpha = self._get_alpha(batch_size)
        x_t = transormation_matrices * (
            x0 + alpha * torch.sqrt(t) * noise)
        
        return x_t.view(* original_batch_shape)

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
        alpha = self._get_alpha(batch_size)
        coefficient_mu_z = alpha / torch.sqrt(t).view(batch_size, -1)
        m_t = self._forward_sample(None, t).view(batch_size, -1)
        m_t_minus_1 = self._forward_sample(None, t - 1).view(batch_size, -1)
        if t_index == 1:
            m_t_minus_1 = 1 + 0 * m_t_minus_1  # identity
        z = self.model(xt, t).view(batch_size, -1)
        xt = xt.view(batch_size, -1)
        model_mean = m_t_minus_1 * ((1 / m_t) * xt - coefficient_mu_z * z)
        x_t_minus_1 = model_mean

        if not deterministic:
            variance_scale = alpha * torch.sqrt(
                (t - 1) / t).view(batch_size, -1)
            noise = torch.randn_like(model_mean)
            x_t_minus_1 += variance_scale * noise * m_t_minus_1
        
        return x_t_minus_1.view(* original_batch_shape)

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            x = torch.randn(shape, device=self.device)
        xs = []

        for t in reversed(range(1, 1 + self.timesteps)):
            x = self.p_sample(x, t, deterministic=deterministic)
            xs.append(x.cpu().numpy())
        return xs

    def _compute_prior_kl_divergence(self, x0, batch_size):
        m_T = self._forward_sample(
            x0, torch.tensor([self.timesteps] * batch_size,
                             device=self.device)).view(batch_size, -1)
        
        # constant = (self._get_alpha(batch_size) ** 2) * self.timesteps
        # trace = (constant * m_T ** 2).sum(dim=1).mean()
        mu_squared = (
            (m_T * x0.view(batch_size, -1)) ** 2).sum(dim=1).mean()
        # log_determinant = torch.log(constant * m_T ** 2).sum(dim=1).mean()
        return 0.5 * (mu_squared - 784)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        t = t + 1  # t \in {1, \dots, timesteps}
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        return loss_weights * noise_loss + kl_divergence / self.timesteps, {
            'alpha': self._get_alpha(batch_size).detach().cpu().numpy(),
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
