import torch
import torch.nn as nn
from torch.distributions import Normal

from .representation import TowerRepresentation
from .generator import GeneratorNetwork


class GenerativeQueryNetwork(nn.Module):
    """
    Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param x_dim: number of channels in input
    :param v_dim: dimensions of viewpoint
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param L: Number of refinements of density
    """
    def __init__(self, x_dim, v_dim, r_dim, h_dim, z_dim, L=12):
        super(GenerativeQueryNetwork, self).__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(x_dim, v_dim, r_dim, z_dim, h_dim, L)
        self.representation = TowerRepresentation(x_dim, v_dim, r_dim, pool=True)

    def forward(self, context_x, context_v, query_x, query_v):
        """
        Forward through the GQN.

        :param x: batch of context images [b, m, c, h, w]
        :param v: batch of context viewpoints for image [b, m, k]
        :param x_q: batch of query images [b, c, h, w]
        :param v_q: batch of query viewpoints [b, k]
        """
        # Merge batch and view dimensions.
        b, m, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        # representation generated from input images
        # and corresponding viewpoints
        phi = self.representation(x, v)

        # Seperate batch and view dimensions
        _, *phi_dims = phi.shape
        phi = phi.view((b, m, *phi_dims))

        # sum over view representations
        r = torch.sum(phi, dim=1)

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(query_x, query_v, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return (x_mu, r, kl)

    def sample(self, context_x, context_v, query_v, sigma):
        """
        Sample from the network given some context and viewpoint.

        :param context_x: set of context images to generate representation
        :param context_v: viewpoints of `context_x`
        :param viewpoint: viewpoint to generate image from
        :param sigma: pixel variance
        """
        batch_size, n_views, _, h, w = context_x.shape

        _, _, *x_dims = context_x.shape
        _, _, *v_dims = context_v.shape

        x = context_x.view((-1, *x_dims))
        v = context_v.view((-1, *v_dims))

        phi = self.representation(x, v)

        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))

        r = torch.sum(phi, dim=1)

        x_mu = self.generator.sample((h, w), query_v, r)
        return x_mu
