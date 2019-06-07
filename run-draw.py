from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from draw import DRAW
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRAW with MNIST Example')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch (default: 64)')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="./train")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)

    args = parser.parse_args()

    # Define dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.bernoulli(x))
    ])
    dataset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)

    # Create model and optimizer
    model = DRAW(x_dim=784, h_dim=256, z_dim=16, T=10).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    # Load the dataset
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    loss = nn.BCELoss(reduce=False).to(device)

    for epoch in range(args.epochs):
        for x, _ in tqdm(loader):
            batch_size = x.size(0)

            x = x.view(batch_size, -1).to(device)

            x_hat, kl_divergence = model(x)
            x_hat = torch.sigmoid(x_hat)

            reconstruction = loss(x_hat, x).sum(1)
            kl = kl_divergence.sum(1)
            elbo = torch.mean(reconstruction + kl)

            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            scheduler.step()

            if epoch % 1 == 0:
                print("Loss at step {}: {}".format(epoch, elbo.item()))

                # Not sustainable if not dataparallel
                if type(model) is nn.DataParallel:
                    x_sample = model.module.sample(args.batch_size)
                else:
                    x_sample = model.sample(args.batch_size)

                save_image(x_hat, "reconstruction-{}.jpg".format(epoch))
                save_image(x_sample, "sample-{}.jpg".format(epoch))

            if epoch % 10 == 0:
                torch.save(model, "model-{}.pt".format(epoch))
