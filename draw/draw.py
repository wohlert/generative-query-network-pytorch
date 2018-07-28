import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class BaseAttention(nn.Module):
    """
    No attention module.
    """
    def __init__(self, h_dim, x_dim):
        super(BaseAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.write_head = nn.Linear(h_dim, x_dim)

    def read(self, x, x_hat, h):
        return torch.cat([x, x_hat], dim=1)

    def write(self, x):
        return self.write_head(x)


class FilterBankAttention(BaseAttention):
    def __init__(self, h_dim, x_dim):
        """
        Filter bank attention mechanism described in the paper.
        """
        super(FilterBankAttention, self).__init__(h_dim, x_dim)

    def read(self, x, error, h):
       return NotImplementedError

    def write(self, x):
        return NotImplementedError


class DRAW(nn.Module):
    """
    Deep Recurrent Attentive Writer (DRAW) [Gregor 2015].

    :param x_dim: size of input
    :param h_dim: number of hidden neurons
    :param z_dim: number of latent neurons
    :param T: number of recurrent layers
    """
    def __init__(self, x_dim, h_dim=256, z_dim=10, T=10, attention_module=BaseAttention):
        super(DRAW, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.T = T

        # Returns the distribution parameters
        self.variational = nn.Linear(h_dim, 2*z_dim)
        self.observation = nn.Linear(x_dim, x_dim)

        # Recurrent encoder/decoder models
        self.encoder = nn.LSTMCell(2*x_dim + h_dim, h_dim)
        self.decoder = nn.LSTMCell(z_dim, h_dim)

        # Attention module
        self.attention = attention_module(h_dim, x_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Hidden states (allocate on same device as input)
        h_enc = x.new_zeros((batch_size, self.h_dim))
        h_dec = x.new_zeros((batch_size, self.h_dim))

        # Cell states
        c_enc = x.new_zeros((batch_size, self.h_dim))
        c_dec = x.new_zeros((batch_size, self.h_dim))

        # Prior distribution
        p_mu = x.new_zeros((batch_size, self.z_dim))
        p_std = x.new_ones((batch_size, self.z_dim))
        self.prior = Normal(p_mu, p_std)

        canvas = x.new_zeros((batch_size, self.x_dim))
        kl = 0

        for _ in range(self.T):
            x_hat = x - F.sigmoid(canvas)
            att = self.attention.read(x, x_hat, h_dec)

            # Infer posterior density from hidden state
            h_enc, c_enc = self.encoder(torch.cat([att, h_dec], dim=1), [h_enc, c_enc])

            # Posterior distribution
            q_mu, q_log_std = torch.split(self.variational(h_enc), self.z_dim, dim=1)
            q_std = torch.exp(q_log_std)
            posterior = Normal(q_mu, q_std)

            # Sample from posterior
            z = posterior.rsample()

            # Send representation through decoder
            h_dec, c_dec = self.decoder(z, [h_dec, c_dec])

            # Gather representation
            canvas += self.attention.write(h_dec)

            kl += kl_divergence(posterior, self.prior)

        # Return the reconstruction
        x_mu = self.observation(canvas)
        return [x_mu, kl]

    def sample(self, z=None):
        """
        Generate a sample from the data distribution.

        :param z: latent code, otherwise sample from prior
        """
        z = self.prior.sample() if z is None else z
        batch_size = z.size(0)

        canvas = z.new_zeros((batch_size, self.x_dim))
        h_dec = z.new_zeros((batch_size, self.h_dim))
        c_dec = z.new_zeros((batch_size, self.h_dim))

        for _ in range(self.T):
            h_dec, c_dec = self.decoder(z, [h_dec, c_dec])
            canvas = canvas + self.attention.write(h_dec)

        x_mu = self.observation(canvas)
        return x_mu


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

        forget_gate = F.sigmoid(self.forget(input))
        input_gate  = F.sigmoid(self.input(input))
        output_gate = F.sigmoid(self.output(input))
        state_gate  = F.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * F.tanh(cell)

        return hidden, cell


class ConvolutionalDRAW(nn.Module):
    """
    Convolutional DRAW model described in
    "Towards Conceptual Compression" [Gregor 2016].
    The model consists of a autoregressive density
    estimator using a recurrent convolutional network.

    :param x_dim: number of channels in input
    :param x_shape: tuple representing input image shape
    :param h_dim: number of hidden channels
    :param z_dim: number of channels in latent variable
    :param T: number of recurrent layers
    """
    def __init__(self, x_dim, x_shape=(32, 32), h_dim=256, z_dim=10, T=10):
        super(ConvolutionalDRAW, self).__init__()
        self.x_dim = x_dim
        self.x_shape = x_shape
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.T = T

        # Outputs parameters of distributions
        self.variational = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.prior = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)

        # Analogous to original DRAW model
        self.write_head = nn.Conv2d(h_dim, x_dim*4, kernel_size=1, stride=1, padding=0)
        self.read_head  = nn.Conv2d(x_dim, x_dim, kernel_size=3, stride=2, padding=1)

        # Recurrent encoder/decoder models
        self.encoder = Conv2dLSTMCell(2*x_dim, h_dim, kernel_size=5, stride=2, padding=2)
        self.decoder = Conv2dLSTMCell(z_dim + x_dim, h_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        h, w = self.x_shape
        batch_size = x.size(0)

        # Hidden states (allocate on same device as input)
        h_enc = x.new_zeros((batch_size, self.h_dim, h//2, w//2))
        h_dec = x.new_zeros((batch_size, self.h_dim, h//2, w//2))

        # Cell states
        c_enc = x.new_zeros((batch_size, self.h_dim, h//2, w//2))
        c_dec = x.new_zeros((batch_size, self.h_dim, h//2, w//2))

        canvas = x.new_zeros((batch_size, self.x_dim, h, w))
        kl = 0

        for _ in range(self.T):
            # Reconstruction error
            epsilon = x - canvas

            # Infer posterior density from hidden state
            h_enc, c_enc = self.encoder(torch.cat([x, epsilon], dim=1), [h_enc, c_enc])

            # Prior distribution
            p_mu, p_log_std = torch.split(self.prior(h_dec), self.z_dim, dim=1)
            p_std = torch.exp(p_log_std)
            prior = Normal(p_mu, p_std)

            # Posterior distribution
            q_mu, q_log_std = torch.split(self.variational(h_enc), self.z_dim, dim=1)
            q_std = torch.exp(q_log_std)
            posterior = Normal(q_mu, q_std)

            # Sample from posterior
            z = posterior.rsample()

            canvas_next = self.read_head(canvas)

            # Send representation through decoder
            h_dec, c_dec = self.decoder(torch.cat([z, canvas_next], dim=1), [h_dec, c_dec])

            # Refine representation
            canvas = canvas + F.pixel_shuffle(self.write_head(h_dec), 2)
            kl += kl_divergence(posterior, prior)

        # Return the reconstruction and kl
        return [canvas, kl]

    def sample(self, x):
        """
        Sample from the prior to generate a new
        datapoint.

        :param x: tensor representing shape of sample
        """
        h, w = self.x_shape
        batch_size = x.size(0)

        h_dec = x.new_zeros((batch_size, self.h_dim, h//2, w//2))
        c_dec = x.new_zeros((batch_size, self.h_dim, h//2, w//2))

        canvas = x.new_zeros((batch_size, self.x_dim, h, w))

        for _ in range(self.T):
            p_mu, p_log_std = torch.split(self.prior(h_dec), self.z_dim, dim=1)
            p_std = torch.exp(p_log_std)
            z = Normal(p_mu, p_std).sample()

            canvas_next  = self.read_head(canvas)
            h_dec, c_dec = self.decoder(torch.cat([z, canvas_next], dim=1), [h_dec, c_dec])
            canvas = canvas + F.pixel_shuffle(self.write_head(h_dec), 2)

        return canvas