"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .gaussian import GaussianDiffusion


def eigen_value_decomposition(matrix):
    matrix = np.array(matrix)
    return np.linalg.eigh(matrix)


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


class Blurring(GaussianDiffusion):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, forward_matrix, reverse_model, timesteps,
        img_shape, fixed_blur=False, schedule='cosine', device='cpu'):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device)
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape
        self.base_blur_matrix = torch.tensor(
            conv_to_dense(gaussian_kernel(sigma=0.35), self.img_dim),
            device=self.device,
            dtype=torch.float32)
        self.reverse_model = reverse_model
        
        self.fixed_blur = fixed_blur
        print('Using fixed blur', fixed_blur)
        eigen_values, eigen_vectors = eigen_value_decomposition(
            self.base_blur_matrix.detach().cpu().numpy())
        self.blur_eigen_values = torch.tensor(
            eigen_values, device=self.device, dtype=torch.float32)
        self.blur_eigen_vectors = torch.tensor(
            eigen_vectors, device=self.device, dtype=torch.float32)
        self.all_blur_eigen_values = torch.tensor(
            [eigen_values ** i for i in range(1, timesteps + 1)],
            device=self.device,
            dtype=torch.float32)

    @torch.no_grad()
    def sample(self, batch_size, x=None, deterministic=False):
        """Samples from the diffusion process, producing images from noise

        Repeatedly takes samples from p(x_{t-1}|x_t) for each t
        """
        shape = (batch_size, self.img_channels, self.img_dim, self.img_dim)
        if x is None: 
            noise = torch.randn(shape, device=self.device)
            t = torch.tensor([self.timesteps - 1] * batch_size,
                             device=self.device)
            transformation_matrices = self._forward_sample(None, t)
            x = torch.bmm(transformation_matrices,
                          noise.view(batch_size, self.img_dim ** 2, 1)).view(* shape)
        xs = [x.cpu().numpy()]

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs

    def _construct_blur_matrix(self, eigenvalues):
        batch_size = eigenvalues.shape[0]
        eigenvalues = eigenvalues.view(batch_size, self.img_dim ** 2, 1)
        return (self.blur_eigen_vectors @
                (eigenvalues * self.blur_eigen_vectors.t())).view(
                    batch_size, self.img_dim ** 2, self.img_dim ** 2)

    def _forward_eigenvalues(self, x0, time):
        if self.fixed_blur:
            return self.all_blur_eigen_values[time]
        return self.forward_matrix(self.model.time_mlp(time))

    def _forward_sample(self, x0, time):
        eigenvalues = self._forward_eigenvalues(x0, time)
        return self._construct_blur_matrix(eigenvalues)

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transformation_matrices = self._forward_sample(x0, t)
        noisy_images = self._add_noise(
            x0, t, noise).view(batch_size, self.img_dim ** 2, 1)
        x_t = torch.bmm(transformation_matrices, noisy_images)
        return x_t.view(* original_batch_shape)

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
            - betas_t * torch.bmm(m_t_minus_1_bar, z) / sqrt_one_minus_bar_alphas_t)

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        batch_size = xt.shape[0]
        original_batch_shape = xt.shape

        t =  torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
        xt_reversed = self.reverse_model(xt, t).view(batch_size, self.img_dim ** 2, 1)
        xt = xt.view(batch_size, self.img_dim ** 2, 1)

        # compute m matrices, shape (batch_size, img_dim ** 2, img_dim ** 2)
        m_t_bar = self._forward_sample(None, t)
        m_t_minus_1_bar = self._forward_sample(None, t - 1)
    
        if t_index == 0:
            m_t_minus_1_bar = torch.eye(
                self.img_dim ** 2,
                device=self.device,
                dtype=m_t_minus_1_bar.dtype)[None, :, :] + 0 * m_t_minus_1_bar  # identity
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        z = self.model(xt.view(* original_batch_shape), t).view(batch_size, self.img_dim ** 2, 1)
        xt_prev_mean = self.p_sample_prev_mean(
            m_inverse_xt=(xt - torch.bmm(m_t_bar, xt_reversed)
                          + torch.bmm(m_t_minus_1_bar, xt_reversed)),
            m_t_minus_1_bar=m_t_minus_1_bar,
            t=t,
            z=z,
            shape=xt.shape)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(self.posterior_variance, t, xt_prev_mean.shape)
            noise = torch.randn_like(xt_prev_mean)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + torch.sqrt(post_var_t) * torch.bmm(m_t_minus_1_bar, noise)

        # return x_{t-1}
        return xt_prev.view(* original_batch_shape)

    def _compute_prior_kl_divergence(self, x0, batch_size):
        t = torch.tensor([self.timesteps - 1] * batch_size,
                         device=self.device)
        epsilon = 1e-3
        eig_T = epsilon + self._forward_eigenvalues(None, t).view(batch_size, -1)
        blur_eigs = epsilon + self.all_blur_eigen_values[-1][None, :]
        trace = ((eig_T / blur_eigs) ** 2).sum(dim=1).mean()
        mu_squared = 0
        log_determinant = 2 * (
            torch.log(eig_T).sum(dim=1).mean()
            - torch.log(blur_eigs).sum())
        # print(trace, log_determinant)
        return 0.5 * (trace + mu_squared - log_determinant - 784)

    def loss_at_step_t(self, x0, t, loss_weights, loss_type='l1', noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            raise NotImplementedError()
        batch_size = x0.shape[0]
        x_noisy = self.q_sample(x0=x0, t=t, noise=noise)
        
        predicted_noise = self.model(x_noisy, t)
        noise_loss = self.p_loss_at_step_t(noise, predicted_noise, loss_type)

        # reverse model loss
        reverse_model_loss = self.p_loss_at_step_t(
            self._add_noise(x0, t, noise).view(batch_size, -1),
            # TODO: detach x_noisy
            self.reverse_model(x_noisy, t).view(batch_size, -1),
            loss_type)

        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)
        total_loss = (reverse_model_loss
                      + loss_weights * noise_loss
                      + kl_divergence / self.timesteps)
        return total_loss, {
            'reverse_model_loss': reverse_model_loss.item(),
            'noise_loss': noise_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
