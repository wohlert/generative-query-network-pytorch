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
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gqn import GenerativeQueryNetwork
from shepardmetzler import ShepardMetzler, Scene

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="./train")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--fp16', type=bool, help='whether to use FP16 (default: False)', default=False)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)

    args = parser.parse_args()

    dataset = ShepardMetzler(root_dir=args.data_dir, transform=None)
    loss = nn.MSELoss(reduce=False).to(device)

    # Pixel variance
    sigma_f, sigma_i = 0.7, 2.0

    # Learning rate
    mu_f, mu_i = 5*10**(-5), 5*10**(-4)
    mu, sigma = mu_f, sigma_f

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=5, r_dim=256, h_dim=128, z_dim=64, L=12).to(device)

    # Model optimisations
    model = nn.DataParallel(model) if args.data_parallel else model
    model = model.half() if args.fp16 else model

    optimizer = torch.optim.Adam(model.parameters(), lr=mu)

    # Load the dataset
    batch_size = args.batch_size
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Perform 2 million gradient updates
    gradient_steps, max_steps = 0, 2*10**6
    
    for epoch in range(args.epochs):
        if gradient_steps >= max_steps:
            torch.save(model, "model-final.pt")
            break

        for x, v in tqdm(loader):
            if args.fp16:
                x, v = x.half(), v.half()

            x = x.to(device)
            v = v.to(device)

            # Sample random number of views
            batch_size, M, *_ = v.size()
            m = torch.multinomial(torch.arange(M), random.randint(4, M))

            x = x.transpose(1, 0).contiguous()[m]
            v = v.transpose(1, 0).contiguous()[m]

            x_hat, x_q, r, kl = model(x, v, sigma)

            # If more than one GPU we must take new shape into account
            batch_size = x_q.size(0)
            
            reconstruction = torch.sum(loss(x_hat, x_q).view(batch_size, -1), dim=1)
            kl_divergence = torch.sum(kl.view(batch_size, -1), dim=1)
            elbo = torch.mean(reconstruction + kl_divergence)

            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

            gradient_steps += 1
        
        with torch.no_grad():
            if epoch % 1 == 0:
                print("Loss at step {}: {}".format(epoch, elbo.item()))

                x, v = next(iter(loader))
                x_hat, _, r, _ = model(x, v, sigma)

                r = r.view(-1, 1, 16, 16)

                save_image(r.float(), "representation-{}.jpg".format(epoch))
                save_image(x_hat.float(), "reconstruction-{}.jpg".format(epoch))

            if epoch % 10 == 0:
                torch.save(model, "model-{}.pt".format(epoch))

            # Anneal learning rate
            s = epoch + 1
            mu = max(mu_f + (mu_i - mu_f)*(1 - s/(1.6 * 10**6)), mu_f)
            optimizer.lr = mu * math.sqrt(1 - 0.999**s)/(1 - 0.9**s)

            # Anneal pixel variance
            sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - s/(2 * 10**5)), sigma_f)