"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .learned import LearnedGaussianDiffusion


def gaussian_kernel(l=5, sigma=1.):
    """Creates gaussian kernel with side length `l` and a sigma of `sig`.
    Taken from:
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    x = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gaussian_1d = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gaussian_1d, gaussian_1d)
    return kernel / np.sum(kernel)


def conv_to_dense(kernel, size):
    weights = []
    kernel_size, _ = kernel.shape
    assert kernel_size % 2 == 1
    kernel_half_width = kernel_size // 2
    base_size = size + 2 * kernel_half_width
    for i in range(size):
        for j in range(size):
            base = np.zeros((base_size, base_size))
            base[i:i + kernel_size, j:j + kernel_size] = kernel
            base = base[kernel_half_width: -kernel_half_width,
                        kernel_half_width: -kernel_half_width]
            weights.append(base.reshape(-1))
    weights = np.asarray(weights)
    # print(weights.shape, base_size)
    assert weights.shape == (size * size, size * size)
    return weights

class LearnedGaussianDiffusionInputTime(LearnedGaussianDiffusion):
    """Implements the core learning and inference algorithms."""

    def __init__(
        self, noise_model, forward_matrix, reverse_model,
        timesteps, img_shape, schedule='linear', device='cpu'):
        super().__init__(
            noise_model, forward_matrix, timesteps, img_shape,
            schedule, device)
        self.reverse_model = reverse_model
        self.epsilon = 1e-6
        self.blur_matrix = torch.tensor(
            conv_to_dense(gaussian_kernel(sigma=5), self.img_dim),
            device=self.device,
            dtype=torch.float32)
        for _ in range(8):
            self.blur_matrix = torch.matmul(
                self.blur_matrix, self.blur_matrix)

    def _clip_output(self, x):
        return self.epsilon + torch.nn.functional.softplus(x)

    def _forward_sample(self, x0, time):
        return self._clip_output(self.forward_matrix(x0, time))

    def _remove_noise(self, x0_noisy, z, t):
        sqrt_bar_alphas_t = get_by_idx(
            self.sqrt_bar_alphas, t, x0_noisy.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, x0_noisy.shape)
        return (x0_noisy
                - sqrt_one_minus_bar_alphas_t * z) / sqrt_bar_alphas_t

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            noise = torch.randn(
                shape, device=self.device)
            t = torch.tensor([self.timesteps - 1] * batch_size,
                             device=self.device)
            transormation_matrices = self._forward_sample(
                noise, t).view(batch_size, -1)
            x = (transormation_matrices * noise.view(batch_size, -1)).view(* shape)
        xs = [x.cpu().numpy()]

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs

    def _compute_prior_kl_divergence(self, x0, batch_size):
        t = torch.tensor([self.timesteps - 1] * batch_size,
                         device=self.device)
        m_T = self._forward_sample(x0, t).view(batch_size, -1)
        blurred_images = torch.matmul(
            x0.view(batch_size, 1, -1),
            self.blur_matrix).view(batch_size, -1)
        return self.p_loss_at_step_t(
            m_T * x0.view(batch_size, -1), blurred_images)

    @torch.no_grad()
    def p_sample_prev_mean(self, m_inverse_xt, m_t_minus_1_bar, z, t, shape):
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            self.sqrt_one_minus_bar_alphas, t, shape)
        betas_t = get_by_idx(self.betas, t, shape)
        return sqrt_recip_alphas_t * (
            # m_t_minus_1_bar * xt / m_t_bar
            m_inverse_xt
            - m_t_minus_1_bar * betas_t * z / sqrt_one_minus_bar_alphas_t)


    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # x0_approx
        z = self.model(xt, t).view(batch_size, -1)
        xt_reversed = self.reverse_model(xt, t).view(batch_size, -1)
        x0_approx = self._remove_noise(xt_reversed, z, t).view(
            * original_batch_shape)
        
        # xt_prev_mean
        m_t_bar = self._forward_sample(x0_approx, t).view(batch_size, -1)
        m_t_minus_1_bar = self._forward_sample(
            x0_approx, t - 1).view(batch_size, -1)
        if t_index == 0:
            m_t_minus_1_bar = 1 + 0 * m_t_minus_1_bar  # identity
        xt = xt.view(batch_size, -1)
        xt_prev_mean = self.p_sample_prev_mean(
            m_inverse_xt=(
                xt - m_t_bar * xt_reversed
                + m_t_minus_1_bar * xt_reversed),
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=z,
            shape=xt.shape)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + m_t_minus_1_bar * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        # noise loss
        predicted_noise = self.model(x_noisy, t)
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        
        # prior kl divergence
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        # reverse model loss
        reverse_model_loss = self.p_loss_at_step_t(
            self._add_noise(x0, t, noise).view(batch_size, -1),
            self.reverse_model(x_noisy.detach(), t).view(batch_size, -1),
            loss_type)

        total_loss = (reverse_model_loss
                      + loss_weights * noise_loss
                      + kl_divergence / self.timesteps)

        return total_loss, {
            'reverse_model_loss': reverse_model_loss.item(),
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }


class InputTimeReparam2(LearnedGaussianDiffusionInputTime):

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # xt_prev_mean
        m_t_bar = self._clip_output(
            self.reverse_model(xt, t).view(batch_size, -1))
        m_t_minus_1_bar = self._clip_output(
            self.reverse_model(xt, t - 1).view(batch_size, -1))
        if t_index == 0:
            m_t_minus_1_bar = 1 + 0 * m_t_minus_1_bar  # identity
        xt = xt.view(batch_size, -1)
        xt_prev_mean = self.p_sample_prev_mean(
            m_inverse_xt=m_t_minus_1_bar * xt / m_t_bar,
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=self.model(xt.view(* original_batch_shape), t).view(batch_size, -1),
            shape=xt.shape)
        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + m_t_minus_1_bar * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        # noise loss
        predicted_noise = self.model(x_noisy, t)
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        
        # prior kl divergence
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        # reverse model loss
        reverse_model_loss = self.p_loss_at_step_t(
            self._forward_sample(x0, t).view(batch_size, -1).detach(),
            self._clip_output(
                self.reverse_model(x_noisy.detach(), t).view(batch_size, -1)),
            loss_type)

        total_loss = (reverse_model_loss
                      + loss_weights * noise_loss
                      + kl_divergence / self.timesteps)

        return total_loss, {
            'reverse_model_loss': reverse_model_loss.item(),
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }


class InputTimeReparam3(LearnedGaussianDiffusionInputTime):

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # xt_prev_mean
        mu, sigma = self.model(xt, t)
        xt = xt.view(batch_size, -1)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)
        xt_prev_mean = sqrt_recip_alphas_t * mu.view(batch_size, -1)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + self._clip_output(sigma).view(
                batch_size, -1) * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        # noise loss
        mu, sigma = self.model(x_noisy, t)
        m_t_bar = self._forward_sample(x0, t).view(batch_size, -1)
        mask = (t > 0).type(t.type()).view(batch_size, -1)
        m_t_minus_1_bar = (1 - mask) + mask * self._forward_sample(x0, t - 1).view(batch_size, -1)
        x_noisy = x_noisy.view(batch_size, -1)
        prev_mu = self.p_sample_prev_mean(
            m_inverse_xt=m_t_minus_1_bar * x_noisy / m_t_bar,
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=noise.view(batch_size, -1),
            shape=x_noisy.shape)
        sqrt_recip_alphas_t = get_by_idx(
            torch.sqrt(1.0 / self.alphas), t, prev_mu.shape)
        mu_loss = self.p_loss_at_step_t(
            mu.view(batch_size, -1),
            prev_mu / sqrt_recip_alphas_t, loss_type)
        sigma_loss = self.p_loss_at_step_t(
            self._clip_output(sigma).view(batch_size, -1),
            m_t_minus_1_bar, loss_type)

        # prior kl divergence
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        total_loss = (mu_loss + sigma_loss
                      + kl_divergence / self.timesteps)

        return total_loss, {
            'mu_loss': mu_loss.item(),
            'sigma_loss': sigma_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }


class InputTimeReparam4(LearnedGaussianDiffusionInputTime):

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # xt_prev_mean
        mu, sigma = self.model(xt, t)
        mu = mu + xt
        xt = xt.view(batch_size, -1)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recip_alphas_t = get_by_idx(sqrt_recip_alphas, t, xt.shape)
        xt_prev_mean = sqrt_recip_alphas_t * mu.view(batch_size, -1)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + self._clip_output(sigma).view(
                batch_size, -1) * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        # noise loss
        mu, sigma = self.model(x_noisy, t)
        m_t_bar = self._forward_sample(x0, t).view(batch_size, -1)
        mask = (t > 0).type(t.type()).view(batch_size, -1)
        m_t_minus_1_bar = (1 - mask) + mask * self._forward_sample(x0, t - 1).view(batch_size, -1)
        x_noisy = x_noisy.view(batch_size, -1)
        prev_mu = self.p_sample_prev_mean(
            m_inverse_xt=m_t_minus_1_bar * x_noisy / m_t_bar,
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=noise.view(batch_size, -1),
            shape=x_noisy.shape)
        sqrt_recip_alphas_t = get_by_idx(
            torch.sqrt(1.0 / self.alphas), t, prev_mu.shape)
        mu_loss = self.p_loss_at_step_t(
            mu.view(batch_size, -1),
            prev_mu / sqrt_recip_alphas_t - x_noisy,
            loss_type)
        sigma_loss = self.p_loss_at_step_t(
            self._clip_output(sigma).view(batch_size, -1),
            m_t_minus_1_bar, loss_type)

        # prior kl divergence
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        total_loss = (mu_loss + sigma_loss
                      + kl_divergence / self.timesteps)

        return total_loss, {
            'mu_loss': mu_loss.item(),
            'sigma_loss': sigma_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }


class InputTimeReparam5(LearnedGaussianDiffusionInputTime):

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        noisy_image = self._add_noise(x0, t, noise)
        transormation_matrices = self._forward_sample(
            noisy_image, t).view(batch_size, -1)
        x_t = transormation_matrices * noisy_image.view(batch_size, -1)
        return x_t.view(* original_batch_shape)

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # xt_prev_mean
        xt_reversed = self.reverse_model(xt, t)
        m_t_bar = self._forward_sample(
            xt_reversed, t).view(batch_size, -1)
        m_t_minus_1_bar = self._forward_sample(
            xt_reversed, t - 1).view(batch_size, -1)
        if t_index == 0:
            m_t_minus_1_bar = 1 + 0 * m_t_minus_1_bar  # identity
        xt = xt.view(batch_size, -1)
        xt_prev_mean = self.p_sample_prev_mean(
            m_inverse_xt=m_t_minus_1_bar * xt / m_t_bar,
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=self.model(xt.view(* original_batch_shape), t).view(batch_size, -1),
            shape=xt.shape)
        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + m_t_minus_1_bar * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is not None: raise NotImplementedError()
        # encode x0 into auxiliary encoder variable a
        if noise is None:
            noise = torch.randn_like(x0)
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)

        # noise loss
        predicted_noise = self.model(x_noisy, t)
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)
        
        # prior kl divergence
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)

        # reverse model loss
        reverse_model_loss = self.p_loss_at_step_t(
            self._add_noise(x0, t, noise),
            self.reverse_model(x_noisy.detach(), t),
            loss_type)

        total_loss = (reverse_model_loss
                      + loss_weights * noise_loss
                      + kl_divergence / self.timesteps)

        return total_loss, {
            'reverse_model_loss': reverse_model_loss.item(),
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }



class InputTimeReparam6(InputTimeReparam5):

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device,
                        dtype=torch.long)
        
        # xt_prev_mean
        xt_reversed = self.reverse_model(xt, t)
        m_t_bar = self._forward_sample(
            xt_reversed, t).view(batch_size, -1)
        m_t_minus_1_bar = self._forward_sample(
            xt_reversed, t - 1).view(batch_size, -1)
        if t_index == 0:
            m_t_minus_1_bar = 1 + 0 * m_t_minus_1_bar  # identity
        xt = xt.view(batch_size, -1)
        xt_reversed = xt_reversed.view(batch_size, -1)
        xt_prev_mean = self.p_sample_prev_mean(
            m_inverse_xt=(
                xt - m_t_bar * xt_reversed + m_t_minus_1_bar * xt_reversed),
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=self.model(xt.view(* original_batch_shape), t).view(batch_size, -1),
            shape=xt.shape)
        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + m_t_minus_1_bar * torch.sqrt(post_var_t) * noise

        return xt_prev.view(* original_batch_shape)