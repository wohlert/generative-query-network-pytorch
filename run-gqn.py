"""
run-gqn.py

Script to train the a GQN on the Shepard-Metzler dataset
in accordance to the hyperparamter settings described in
the supplementary materials of the paper.
"""
import sys
import random
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gqn import GenerativeQueryNetwork
from shepardmetzler import ShepardMetzler, Scene, transform_viewpoint

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--gradient_steps', type=int, default=2*(10**6), help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="train")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--fp16', type=bool, help='whether to use FP16 (default: False)', default=False)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)

    args = parser.parse_args()

    dataset = ShepardMetzler(root_dir=args.data_dir, target_transform=transform_viewpoint)

    # Pixel variance
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    mu, sigma = mu_f, sigma_f

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12).to(device)

    # Model optimisations
    model = nn.DataParallel(model) if args.data_parallel else model
    model = model.half() if args.fp16 else model

    optimizer = torch.optim.Adam(model.parameters(), lr=mu)

    # Load the dataset
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Number of gradient steps
    s = 0
    while True:
        if s >= args.gradient_steps:
            torch.save(model, "model-final.pt")
            break

        for x, v in tqdm(loader):
            if args.fp16:
                x, v = x.half(), v.half()

            x = x.to(device)
            v = v.to(device)

            x_mu, x_q, r, kld = model(x, v)

            # If more than one GPU we must take new shape into account
            batch_size = x_q.size(0)

            # Negative log likelihood
            nll = -Normal(x_mu, sigma).log_prob(x_q)

            reconstruction = torch.mean(nll.view(batch_size, -1), dim=0).sum()
            kl_divergence  = torch.mean(kld.view(batch_size, -1), dim=0).sum()

            # Evidence lower bound
            elbo = reconstruction + kl_divergence
            elbo.backward()

            optimizer.step()
            optimizer.zero_grad()

            s += 1

            # Keep a checkpoint every 100,000 steps
            if s % 100000 == 0:
                torch.save(model, "model-{}.pt".format(s))

        with torch.no_grad():
            print("|Steps: {}\t|NLL: {}\t|KL: {}\t|".format(s, reconstruction.item(), kl_divergence.item()))

            x, v = next(iter(loader))
            x, v = x.to(device), v.to(device)

            x_mu, _, r, _ = model(x, v)

            r = r.view(-1, 1, 16, 16)

            save_image(r.float(), "representation.jpg")
            save_image(x_mu.float(), "reconstruction.jpg")

            # Anneal learning rate
            mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999**s)/(1 - 0.9**s)

            # Anneal pixel variance
            sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)
