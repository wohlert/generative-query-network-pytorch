"""
The inference-generator architecture is conceptually
similar to the encoder-decoder pair seen in variational
autoencoders. The difference here is that the model
must infer latents from a cascade of time-dependent inputs
using convolutional and recurrent networks.

Additionally, a representation vector is shared between
the networks.
"""
SCALE = 4 # Scale of image generation process


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class Conv2dLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.
    Functionally equivalent to nn.LSTMCell with the
    difference being that nn.Kinear layers are replaced
    by nn.Conv2D layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of image kernel
    :param stride: length of kernel stride
    :param padding: number of pixels to pad with
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        """
        Send input through the cell.

        :param input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class GeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: number of density refinements
    :param share: whether to share cores across refinements
    """
    def __init__(self, x_dim, v_dim, r_dim, z_dim=64, h_dim=128, L=12, share=True):
        super(GeneratorNetwork, self).__init__()
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.share = share

        # Core computational units
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        inference_args = dict(in_channels=v_dim + r_dim + x_dim + h_dim, out_channels=h_dim, **kwargs)
        generator_args = dict(in_channels=v_dim + r_dim + z_dim, out_channels=h_dim, **kwargs)
        if self.share:
            self.inference_core = Conv2dLSTMCell(**inference_args)
            self.generator_core = Conv2dLSTMCell(**generator_args)
        else:
            self.inference_core = nn.ModuleList([Conv2dLSTMCell(**inference_args) for _ in range(L)])
            self.generator_core = nn.ModuleList([Conv2dLSTMCell(**generator_args) for _ in range(L)])

        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, **kwargs)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, **kwargs)

        # Generative density
        self.observation_density = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)

        # Up/down-sampling primitives
        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=SCALE, stride=SCALE, padding=0, bias=False)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=SCALE, stride=SCALE, padding=0, bias=False)

    def forward(self, x, v, r):
        """
        Attempt to reconstruct x with corresponding
        viewpoint v and context representation r.

        :param x: image to send through
        :param v: viewpoint of image
        :param r: representation for image
        :return reconstruction of x and kl-divergence
        """
        batch_size, _, h, w = x.shape
        kl = 0

        # Downsample x, upsample v and r
        x = self.downsample(x)
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)

        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_i   = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        hidden_g = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_g   = x.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        # Canvas for updating
        u = x.new_zeros((batch_size, self.h_dim, h, w))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))

            # Inference state update
            inference = self.inference_core if self.share else self.inference_core[l]
            hidden_i, cell_i = inference(torch.cat([hidden_g, x, v, r], dim=1), [hidden_i, cell_i])

            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))

            # Posterior sample
            z = posterior_distribution.rsample()

            # Generator state update
            generator = self.generator_core if self.share else self.generator_core[l]
            hidden_g, cell_g = generator(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])

            # Calculate u
            u = self.upsample(hidden_g) + u

            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)

        x_mu = self.observation_density(u)

        return torch.sigmoid(x_mu), kl

    def sample(self, x_shape, v, r):
        """
        Sample from the prior distribution to generate
        a new image given a viewpoint and representation

        :param x_shape: (height, width) of image
        :param v: viewpoint
        :param r: representation (context)
        """
        h, w = x_shape
        batch_size = v.size(0)

        # Increase dimensions
        v = v.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)

        # Reset hidden and cell state for generator
        hidden_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))

        u = v.new_zeros((batch_size, self.h_dim, h, w))

        for _ in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))

            # Prior sample
            z = prior_distribution.sample()

            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u

        x_mu = self.observation_density(u)
        
        return torch.sigmoid(x_mu)
