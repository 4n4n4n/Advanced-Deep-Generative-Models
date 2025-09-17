from functools import partial

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


# A batch of (noisy) images
ImageBatch = TensorType["batch_size", "channels", "height", "width", torch.float32]

# Integer noise level between 0 and N - 1
NoiseLevel = TensorType["batch_size", torch.long]

# Normalized noise level between 0 and 1
NormalizedNoiseLevel = TensorType["batch_size", torch.float32]


def batch_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Make `a` broadcast along the batch dimension of `b`.

    We assume the batch dimension to be the first one.
    """

    assert a.ndim == 1
    return a.view(-1, *((1,) * (b.ndim - 1)))


class ResNet(nn.Module):
    """A minimal convolutional residual network."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    ConvLayer(hidden_dim, hidden_dim, kernel_size=3),
                )
                for i in range(n_layers)
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None].expand(n.shape[0], -1, *z_n.shape[2:])

        z_n = self.embed(torch.cat((z_n, n), dim=-3))

        for layer in self.layers:
            z_n = z_n + layer(torch.cat((z_n, n), dim=-3))

        return self.out(z_n)


class MiniUnet(nn.Module):
    """A minimal U-net implementation [1].

    [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox: "U-Net: Convolutional Networks
        for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        assert n_layers <= 2, (
            "MNIST images can only be downsampled twice "
            "without taking care of padding issues"
        )

        self.n_layers = n_layers

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        # At each scale, we perform one nonlinear map with residual connection
        self.downscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in range(n_layers)
            ]
        )
        bottom_channels = 4**n_layers * hidden_dim
        self.bottom_map = nn.Sequential(
            ConvLayer(bottom_channels + 1, bottom_channels),
            nn.SiLU(),
            ConvLayer(bottom_channels, bottom_channels),
        )
        self.upscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(2 * 4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in reversed(range(1, n_layers + 1))
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None]

        def cat_n(z_n, *tensors):
            return torch.cat((z_n, *tensors, n.expand(-1, -1, *z_n.shape[2:])), dim=-3)

        z_n = self.embed(cat_n(z_n))

        skip_connections = []
        for down_layer in self.downscaling:
            z_n = z_n + down_layer(cat_n(z_n))
            z_n = eo.rearrange(z_n, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=2, w2=2)
            skip_connections.append(z_n)

        z_n = self.bottom_map(cat_n(z_n))

        for up_layer in self.upscaling:
            z_n = z_n + up_layer(cat_n(z_n, skip_connections.pop()))
            z_n = eo.rearrange(z_n, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)

        return self.out(z_n)


class DDPM(nn.Module):
    """A denoising diffusion model as described in [1].

    References:

    [1] "Denoising Diffusion Probabilistic Models", Ho et al., https://arxiv.org/abs/2006.11239
    """

    def __init__(self, N: int, type: str, hidden_dim: int, n_layers: int):
        """Initialize the diffusion model.

        Args:
            N: Number of diffusion steps
        """

        super().__init__()
        self.N = N
        self.type = type

        if type == "resnet":
            self.model = ResNet(feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        elif type == "unet":
            self.model = MiniUnet(
                feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
            )
        else:
            raise RuntimeError(f"Unknown model type {type}")

        # Compute a beta schedule and various derived variables as defined on the slides
        ##########################################################
        # Beta schedule from the DDPM paper, linearly increasing from small to large noise
        beta = torch.linspace(1e-4, 2e-2, N)  
        
        # Alpha values are 1 - beta
        alpha = 1.0 - beta

        # Cumulative product of alphas
        alpha_bar = torch.zeros_like(alpha)
        alpha_bar[0] = alpha[0]
        for i in range(1,N):
            alpha_bar[i] = alpha_bar[i-1]*alpha[i]
        
        # Calculate the variance of the posterior q(z_{n-1}|z_n, x_0)
        beta_tilde = torch.zeros_like(beta)
        beta_tilde[0] = 0
        for n in range(1, N):
                beta_tilde[n] = beta[n] * (1 - alpha_bar[n - 1]) / (1 - alpha_bar[n])

        ##########################################################
        # Register the parameters as buffer tensors to make them accessible
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha", alpha.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("beta_tilde", beta_tilde.float())
    @typechecked
    def simplified_loss(
        self, x0: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> torch.Tensor:
        """Compute the simplified ELBO loss.

        Args:
            x0: Raw image data to compute the loss for
            n: Noise level
            epsilon: Noise instance

        Returns:
            0-dimensional tensor of the fully-reduced loss
        """

        ##########################################################
        # YOUR CODE HERE
        broad_alpha_bar = batch_broadcast(self.alpha_bar[n], x0)
        zn = torch.sqrt(broad_alpha_bar)*x0 + torch.sqrt(1-broad_alpha_bar)*epsilon
        pred_epsilon = self.model(zn, n/self.N)
        
        # ImageBatch
        loss = torch.sum(torch.square(pred_epsilon - epsilon))/x0.shape[0]
        # print(loss)
        return loss
        ##########################################################

    def loss(self, x0: ImageBatch) -> torch.Tensor:
        batch_size = x0.shape[0]
        n = torch.randint(self.N, (batch_size,), device=x0.device)
        epsilon = torch.randn_like(x0)

        return self.simplified_loss(x0, n, epsilon)

    @typechecked
    def estimate_x0(
        self, z_n: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> ImageBatch:
        """Re-construct x_0 from z_n and epsilon.

        Args:
            z_n: Noise images
            n: Noise level
            epsilon: Noise that produced z_n

        Returns:
            The reconstructed x_0
        """

        ##########################################################
        # YOUR CODE HERE
        # it's actually from slide 18
        
        # epsilon = (batch, 1,28,28)
        # zn 
        
        # print(z_n.shape, epsilon.shape, self.alpha_bar.shape, self.alpha_bar[n].shape)
        # (20, 1, 28, 28) (20, 1, 28, 28) (20, 1, 1, 1)
        broad_alpha_bar = batch_broadcast(self.alpha_bar[n], epsilon)
        return (z_n - epsilon * torch.sqrt(1-broad_alpha_bar))/torch.sqrt(broad_alpha_bar)
        ##########################################################

    @typechecked
    def sample_z_n_previous(
        self, x0: ImageBatch, z_n: ImageBatch, n: NoiseLevel
    ) -> ImageBatch:
        """Sample z_{n-1} given z_n and x_0.

        Args:
            x0: (Estimate of) images
            z_n: Noisy images
            n: Noise level

        Returns:
            A z_{n-1} sample
        """

        ##########################################################
        # YOUR CODE HERE
        broad_alpha_n = batch_broadcast(self.alpha[n], z_n)
        broad_alpha_bar_n = batch_broadcast(self.alpha_bar[n], z_n)
        broad_alpha_bar_n_1 = batch_broadcast(self.alpha_bar[n-1], z_n)
        broad_beta_n = batch_broadcast(self.beta[n], z_n) 
        broad_beta_tilde_n = batch_broadcast(self.beta_tilde[n], z_n)       
        
        mu = torch.sqrt(broad_alpha_n) * (1- broad_alpha_bar_n_1) / (1-broad_alpha_bar_n) * z_n + torch.sqrt(broad_alpha_bar_n_1) * broad_beta_n / (1-broad_alpha_bar_n) * x0 
        epsilon = torch.randn_like(mu) 
        return mu + epsilon * torch.sqrt(broad_beta_tilde_n)
        ##########################################################

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> ImageBatch:
        """Sample new images from scratch by iteratively denoising pure noise.

        Args:
            batch_size: Number of images to generate
            device: Device to generate them on

        Returns:
            Generated images
        """

        ##########################################################
        # YOUR CODE HERE
        # Start with pure noise (z_N)
        z_n = torch.randn(batch_size, 1, 28, 28, device=device)
        
        # Iteratively denoise from n=N to n=1
        for n in range(self.N-1, -1, -1):
            n_tensor = torch.full((batch_size,), n, device=device, dtype=torch.long)
            # Predict noise
            epsilon = self.model(z_n, n_tensor/self.N)
            # Estimate x0
            x0_hat = self.estimate_x0(z_n, n_tensor, epsilon)
            # Sample z_{n-1}
            z_n = self.sample_z_n_previous(x0_hat, z_n, n_tensor)
        
        # Return final denoised images
        return z_n
        ##########################################################
