"""Implements the core diffusion algorithms."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from .schedule import get_schedule, get_by_idx
from .learned_blurring import Blurring


class Transform(Blurring):
    """Implements the core learning and inference algorithms."""
    def __init__(
        self, noise_model, forward_matrix, reverse_model, timesteps,
        img_shape, blur_initializer, schedule='cosine',
        device='cpu', drop_forward_coef=False, loss_type='elbo'):
        super().__init__(
            noise_model, timesteps, img_shape, schedule, device)
        self.loss_type = loss_type
        self.drop_forward_coef = drop_forward_coef
        self.blur_no_reparam = blur_no_reparam
        self.forward_matrix = forward_matrix
        self.z_shape = img_shape
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
        self.blur_params = torch.nn.Parameter(
            self._initialize_blur_params(blur_initializer))
        self.identity = torch.eye(
            self.img_dim ** 2,
            dtype=torch.float32,
            device=self.device)[None, :, :]
        self.base_eigen_value = self.all_blur_eigen_values[0]
