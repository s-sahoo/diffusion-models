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
        img_shape, level_initializer, fixed_blur=False, schedule='cosine',
        device='cpu', drop_forward_coef=False, levels_no_reparam=False,
        loss_type='elbo', transform_type='blur', sampler='naive', ub_loss=False,
        detach_matrix=False):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device)
        self.loss_type = loss_type
        self.drop_forward_coef = drop_forward_coef
        self.levels_no_reparam = levels_no_reparam
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape
        self.reverse_model = reverse_model
        self.sampler = sampler
        self.transform_type = transform_type
        self.fixed_blur = fixed_blur
        self.ub_loss = ub_loss
        self.detach_matrix = detach_matrix
        print('Detaching blur matrix from the loss:', self.detach_matrix)
        print('UB loss:', self.ub_loss)
        if self.schedule not in self.ddpm_schedules:
            self.gammas = self._gamma_scheduler()
        if self.transform_type == 'blur':
            self.base_blur_matrix = torch.tensor(
                conv_to_dense(
                    gaussian_kernel(l=2 * (self.img_dim // 2) - 1,
                                    sigma=0.35),
                    self.img_dim),
                device=self.device,
                dtype=torch.float32)
            print('Using fixed blur', fixed_blur)
            eigen_values, eigen_vectors = eigen_value_decomposition(
                self.base_blur_matrix.detach().cpu().numpy())
            self.eigen_values = torch.tensor(
                eigen_values, device=self.device, dtype=torch.float32)
            self.eigen_vectors = torch.tensor(
                eigen_vectors, device=self.device, dtype=torch.float32)
            self.all_eigen_values = torch.tensor(
                [eigen_values ** i for i in range(1, timesteps + 1)],
                device=self.device,
                dtype=torch.float32)
        elif self.transform_type == 'learnable_forward':
            print('Transform: learnable_forward')
            self.eigen_vectors = torch.nn.utils.parametrizations.orthogonal(
                nn.Linear(self.img_dim ** 2, self.img_dim ** 2,
                          device=self.device))
            self.eigen_values = torch.nn.Parameter(
                torch.ones(self.img_dim ** 2, device=self.device))
        elif self.transform_type == 'identity':
            print('Transform: identity')
            self.eigen_vectors = torch.eye(
                self.img_dim ** 2,
                dtype=torch.float32,
                device=self.device)
            self.eigen_values = torch.ones(
                self.img_dim ** 2, device=self.device)
        self.level_initializer = level_initializer
        if self.level_initializer == 'sigmoid':
            print('Using sigmoid blur levels.')
            # w * sigmoid(a * (t / T) + b)
            # blur params are initialized between [0.2, 9.8].
            self.w = torch.nn.Parameter(
                10 * torch.ones(1, device=self.device))
            self.a = torch.nn.Parameter(
                8 * torch.ones(1, device=self.device))
            self.b = torch.nn.Parameter(
                -4 * torch.ones(1, device=self.device))
        else:
            # blur_levels = torch.nn.functional.softplus(self.levels)
            self.levels = torch.nn.Parameter(self._initialize_levels())
        self.identity = torch.eye(
            self.img_dim ** 2,
            dtype=torch.float32,
            device=self.device)[None, :, :]

    def _cosine_schedule(self, start, end, tau):
        v_start = torch.cos(start * torch.pi / 2) ** (2 * tau)
        v_end = torch.cos(end * torch.pi / 2) ** (2 * tau)
        t = torch.flip(self._linear_schedule(), (0,))
        output = torch.cos((t * (end - start) + start) * torch.pi / 2) ** (2 * tau)
        return (v_end - output) / (v_end - v_start)

    def _linear_schedule(self):
        return torch.arange(self.timesteps - 1, -1, -1) / (self.timesteps - 1)

    def _gamma_scheduler(self):
        if self.schedule == 'new_linear':
            gammas = self._linear_schedule()
        elif self.schedule == 'new_cosine_1':
            gammas = self._cosine_schedule(
                start=torch.tensor(0.0),
                end=torch.tensor(1.0),
                tau=torch.tensor(1.0))
        elif self.schedule == 'new_cosine_2':
            gammas = self._cosine_schedule(
                start=torch.tensor(0.2),
                end=torch.tensor(1.0),
                tau=torch.tensor(1.0))
        elif self.schedule == 'new_cosine_3':
            gammas = self._cosine_schedule(
                start=torch.tensor(0.2),
                end=torch.tensor(1.0),
                tau=torch.tensor(2.0))
        elif self.schedule == 'new_cosine_4':
            gammas = self._cosine_schedule(
                start=torch.tensor(0.2),
                end=torch.tensor(1.0),
                tau=torch.tensor(3.0))
        return torch.clip(gammas, 1e-9, 1)

    def _get_noise_sigma(self):
        if self.schedule in self.ddpm_schedules:
            return self.sqrt_one_minus_bar_alphas
        return torch.sqrt(1 - self.gammas)

    def _get_x0_coefficient(self):
        if self.schedule in self.ddpm_schedules:
            if self.drop_forward_coef:
                return torch.ones(self.timesteps, device=self.device)
            return self.sqrt_bar_alphas
        return torch.sqrt(self.gammas)

    def _initialize_levels(self):
        if self.level_initializer == 'random':
            return torch.rand(self.timesteps, device=self.device)
        elif self.level_initializer == 'zero':
            # softplus of -3 is close to 0.
            return -3 * torch.ones(
                self.timesteps, device=self.device)
        elif self.level_initializer == 'linear':
            return torch.arange(
                start=0, end=self.timesteps, device=self.device)

    def _construct_transform_matrix(self, eigen_values):
        batch_size = eigen_values.shape[0]
        if self.transform_type in ['blur', 'identity']:
            eigen_vectors = self.eigen_vectors
        else:
            eigen_vectors = self.eigen_vectors.weight
            eigen_values = torch.nn.functional.softplus(
                self.eigen_values)
        eigen_values = eigen_values.view(batch_size, self.img_dim ** 2, 1)
        return (eigen_vectors @ (eigen_values * eigen_vectors.t())).view(
                    batch_size, self.img_dim ** 2, self.img_dim ** 2)

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
            x = self.p_sample(
                x, t, deterministic=(t==0 or deterministic))
            xs.append(x.cpu().numpy())
        return xs

    def _get_levels(self, time):
        if self.level_initializer == 'sigmoid':
            # w * sigmoid(a * (t / T) + b)
            blur_levels = self.w * torch.sigmoid(
                self.a * (time / self.timesteps) + self.b)
            return blur_levels[:, None]
        blur_levels = torch.nn.functional.softplus(self.levels)
        if not self.levels_no_reparam:
            blur_levels = torch.cumsum(blur_levels, dim=0)
            blur_levels = (blur_levels - self.levels).detach() + self.levels
        return blur_levels[time][:, None]

    def _forward_eigenvalues(self, x0, time):
        if self.fixed_blur:
            return self.all_eigen_values[time]
        levels_t = self._get_levels(time)
        return torch.exp(levels_t * torch.log(
            self.eigen_values[None, :]))

    def _get_transform_matrices(self, x0, time):
        eigenvalues = self._forward_eigenvalues(x0, time)
        return self._construct_transform_matrix(eigenvalues)

    def q_sample(self, x0, t, noise=None):
        """Samples from the forward diffusion process q.

        Takes a sample from q(xt|x0). t \in {1, \dots, timesteps}
        """
        if noise is None:
            noise = torch.randn_like(x0)
        original_batch_shape = x0.shape
        batch_size = original_batch_shape[0]
        transformation_matrices = self._get_transform_matrices(x0, t)
        x0 = x0.view(batch_size, self.img_dim ** 2, 1)
        blurred_images = torch.bmm(
            transformation_matrices, x0).view(
                * original_batch_shape)
        x_t = self._add_noise(blurred_images, t, noise)
        return x_t

    def _momentum_sampler(self, xt, t, t_index, deterministic=False):
        x0_approx = xt + self.reverse_model(xt, t)
        if t_index == 0:
            return x0_approx
        yt_hat = self.q_sample(x0_approx, t, torch.zeros_like(x0_approx))
        yt_minus_1_hat = self.q_sample(
            x0_approx, t - 1, torch.zeros_like(x0_approx))
        
        eta = torch.randn_like(x0_approx)
        
        epsilon = yt_hat - xt
        sigma = get_by_idx(
            self._get_noise_sigma(), t, xt.shape)
        sigma_prev = get_by_idx(
            self._get_noise_sigma(), t - 1, xt.shape)
        z = xt - ((sigma_prev / sigma) ** 2 - 1) * epsilon + torch.sqrt(
            sigma ** 2 - sigma_prev ** 2) * eta
        return z + yt_minus_1_hat - yt_hat

    def _naive_sampler(self, xt, t, t_index, deterministic=False):    
        x0_approx = xt + self.reverse_model(xt, t)
        if t_index == 0:
            return x0_approx
        noise = None
        if deterministic:
            noise = torch.zeros_like(x0_approx)
        return self.q_sample(x0_approx, t - 1, noise)


    def _ddpm_sampler(self, xt, t, t_index, deterministic=False):
        x0_approx = xt + self.reverse_model(xt, t)
        if t_index == 0:
            return x0_approx
        bar_alphas = self.gammas
        bar_alphas_prev = F.pad(bar_alphas[:-1], (1, 0), value=1.0)
        alphas = bar_alphas / bar_alphas_prev
        betas = 1 - alphas
        betas_t = get_by_idx(betas, t, xt.shape)
        sqrt_one_minus_bar_alphas_t = get_by_idx(
            torch.sqrt(1 - bar_alphas), t, xt.shape)
        sqrt_recip_alphas_t = get_by_idx(torch.sqrt(1.0 / alphas), t, xt.shape)
        
        # compute epsilon
        predicted_noise = xt - self.q_sample(
            x0_approx, t, torch.zeros_like(x0_approx))
        
        xt_prev_mean = sqrt_recip_alphas_t * (
            xt - betas_t * predicted_noise / sqrt_one_minus_bar_alphas_t)

        if deterministic:
            xt_prev = xt_prev_mean # need this?
        else:
            post_var_t = get_by_idx(
                betas * (1. - bar_alphas_prev) / (1. - bar_alphas),
                t,
                xt.shape)
            noise = torch.randn_like(xt)
            # Algorithm 2 line 4:
            xt_prev = xt_prev_mean + torch.sqrt(post_var_t) * noise

        # return x_{t-1}
        return xt_prev


    def _naive_sampler_clipped(self, xt, t, t_index, deterministic=False):    
        x0_approx = xt + self.reverse_model(xt, t)
        x0_approx = torch.clip(x0_approx, -1, 1)
        if t_index == 0:
            return x0_approx
        noise = None
        if deterministic:
            noise = torch.zeros_like(x0_approx)
        return self.q_sample(x0_approx, t - 1, noise)

    @torch.no_grad()
    def p_sample(self, xt, t_index, deterministic=False):
        """Samples from the reverse diffusion process p at time step t.

        Generate image x_{t_index  - 1}.
        """
        samplers = {
            'momentum': self._momentum_sampler,
            'naive': self._naive_sampler,
            'naive_clipped': self._naive_sampler_clipped,
            'ddpm': self._ddpm_sampler,
        }
        return samplers[self.sampler](
            xt=xt,
            t=torch.full(
                (xt.shape[0],), t_index, device=self.device,
                dtype=torch.long),
            t_index=t_index,
            deterministic=deterministic)

    def _compute_prior_kl_divergence(self, x0, batch_size):
        transformation_matrices = self._get_effective_transform_matrices(
            x0,
            torch.tensor([self.timesteps - 1] * batch_size,
                device=self.device),
            mask=0)
        mu_squared = torch.bmm(
            transformation_matrices,
            x0.view(batch_size, self.img_dim ** 2, 1))
        mu_squared = (mu_squared ** 2).view(batch_size, -1).sum(dim=1).mean()
        return 0.5 * mu_squared

    def _get_effective_transform_matrices(self, x0, t, mask):
        transformation_matrices = self._get_transform_matrices(x0, t)
        blur_scale = get_by_idx(
            self._get_x0_coefficient(),
            t * (t >= 0),
            transformation_matrices.shape)
        # print(blur_scale)
        return ((1 - mask) * blur_scale * transformation_matrices
                + mask * self.identity)

    def loss_at_step_t(self, x0, t, loss_weights, loss_scale, loss_type, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            raise NotImplementedError()
        batch_size = x0.shape[0]
        x_t = self.q_sample(x0=x0, t=t, noise=noise)

        # print(t)
        if self.loss_type == 'elbo':
            transformation_matrices = self._get_effective_transform_matrices(
                x0, t - 1, mask=(t == 0).type(x0.dtype)[:, None, None])
        elif self.loss_type == 'soft_diffusion':
            transformation_matrices = self._get_effective_transform_matrices(
                x0, t, mask=0)
        
        if self.detach_matrix:
            transformation_matrices = transformation_matrices.detach()

        if self.ub_loss:
            target = (x0 - x_t).view(batch_size, self.img_dim ** 2, 1)
            prediction = self.reverse_model(x_t, t).view(
                batch_size, self.img_dim ** 2, 1)
        else:
            target = torch.bmm(
                transformation_matrices,
                (x0 - x_t).view(batch_size, self.img_dim ** 2, 1))
            prediction = torch.bmm(
                transformation_matrices,
                self.reverse_model(x_t, t).view(batch_size, self.img_dim ** 2, 1))
        loss_scale_t = get_by_idx(loss_scale, t, target.shape)
        reconstruction_loss = self.p_loss_at_step_t(
            target * loss_scale_t, prediction * loss_scale_t, loss_type)
        kl_divergence = self._compute_prior_kl_divergence(x0, batch_size)
        total_loss = reconstruction_loss + kl_divergence / (loss_weights * self.timesteps)

        return total_loss, {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
        }
