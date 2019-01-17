import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN, MNIST
from torchvision.utils import save_image

from draw import ConvolutionalDRAW
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConvolutionalDRAW with MNIST/SVHN Example')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="./train")
    parser.add_argument('--batch_size', type=int, default=128, help='size of batch (default: 128)')
    parser.add_argument('--dataset', type=str, default="MNIST", help='dataset to use (default: MNIST)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)

    args = parser.parse_args()

    if args.dataset == "MNIST":
        mean, std = 0, 1
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.1307,), std=(0.3081,)
            transforms.Lambda(lambda x: torch.bernoulli(x))
        ])
        dataset = MNIST(root=args.data_dir, train=True, download=True, transform=transform)
        loss = nn.BCELoss(reduce=False)
        output_activation = torch.sigmoid
        x_dim, x_shape = 1, (28, 28)

    elif args.dataset == "SVHN":
        mean, std = (0.4376, 0.4437, 0.4728), (0.198, 0.201, 0.197)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = SVHN(root=args.data_dir, split="train", download=True, transform=transform)
        loss = nn.MSELoss(reduce=False)
        output_activation = lambda x: x
        x_dim, x_shape = 3, (32, 32)

    # Create model and optimizer
    model = ConvolutionalDRAW(x_dim=x_dim, x_shape=x_shape, h_dim=160, z_dim=12, T=16).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)

    # Load the dataset
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    for epoch in range(args.epochs):
        for x, _ in tqdm(loader):
            batch_size = x.size(0)
            x = x.to(device)

            x_hat, kl = model(x)
            x_hat = output_activation(x_hat)
            
            reconstruction = torch.sum(loss(x_hat, x).view(batch_size, -1), dim=1)
            kl_divergence = torch.sum(kl.view(batch_size, -1), dim=1)
            elbo = torch.mean(reconstruction + kl_divergence)

            elbo.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        with torch.no_grad():
            scheduler.step()

            if epoch % 1 == 0:
                print("Loss at step {}: {}".format(epoch, elbo.item()))

                # Not sustainable if not dataparallel
                x_sample = model.module.sample(args.batch_size)

                # Renormalize to visualise
                x_sample = (x_sample - mean)/std
                x_hat = (x_hat - mean)/std

                save_image(x_hat, "reconstruction-{}.jpg".format(epoch))
                save_image(x_sample, "sample-{}.jpg".format(epoch))

            if epoch % 10 == 0:
                torch.save(model, "model-{}.pt".format(epoch))