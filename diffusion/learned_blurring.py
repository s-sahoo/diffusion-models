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

    def _construct_blur_matrix(self, eigenvalues):
        batch_size = eigenvalues.shape[0]
        eigenvalues = eigenvalues.view(batch_size, self.img_dim ** 2, 1)
        return (self.blur_eigen_vectors @
                (eigenvalues * self.blur_eigen_vectors.t())).view(
                    batch_size, self.img_dim ** 2, self.img_dim ** 2)

    def _forward_eigenvalues(self, x0, time):
        if self.fixed_blur:
            return self.all_blur_eigen_values[time]
        return self.forward_matrix(self.reverse_model.time_mlp(time))

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
        x0 = x0.view(batch_size, self.img_dim ** 2, 1)
        blurred_images = torch.bmm(
            transformation_matrices, x0).view(
                * original_batch_shape)
        x_t = self._add_noise(blurred_images, t, noise)
        return x_t

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
        x_t = self.q_sample(x0=x0, t=t, noise=noise)

        # reverse model loss
        transformation_matrices = self._forward_sample(x0, t)
        target = torch.bmm(
            transformation_matrices,
            (x0 - x_t).view(batch_size, self.img_dim ** 2, 1))
        prediction = torch.bmm(
            transformation_matrices,
            self.reverse_model(x_t, t).view(batch_size, self.img_dim ** 2, 1))
        reverse_model_loss = self.p_loss_at_step_t(target, prediction, 'l2')

        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)
        total_loss = (reverse_model_loss
                      + kl_divergence / self.timesteps)
        return total_loss, {
            'reverse_model_loss': reverse_model_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
