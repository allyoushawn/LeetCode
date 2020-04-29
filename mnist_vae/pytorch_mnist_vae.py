from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


def plot_mnist(X, nrows=10, ncols=10):
    """
    Plots the given MNIST digits on a grid of size (nrows x ncols).
    """
    N = len(X)
    assert N <= nrows * ncols

    # Plot each 28 x 28 image.
    plt.figure(figsize=(nrows, ncols))
    for i in range(N):
        x = X[i].reshape(28, 28)

        plt.subplot(nrows, ncols, i + 1)
        fig = plt.imshow(x, vmin=0, vmax=1, cmap="gray")

        # Hide axes. (Both x-axis and y-axis range from 0 to 27.)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

    # If you want to save the plot using plt.savefig(fig_path), please
    # follow the instructions above (under "How to save files").



class NeuralNetwork(nn.Module):
    def __init__(self, device, algo='vae', lr=0.001, k=500):
        super(NeuralNetwork, self).__init__()
        assert(algo in ['vae', 'wake-sleep'])
        self.algo = algo
        self.lr = lr
        self.k = k
        self.device = device

        self.z_dim = 2
        self.input_size = 784

        self._create_network()

    def _create_network(self, n_hidden_r=512, n_hidden_g=512):
        self._recognition_network(n_hidden_r)
        self._generator_netowrk(n_hidden_g)

    def _recognition_network(self, n_hidden_r):
        self.x2hid = nn.Linear(self.input_size, n_hidden_r)
        self.hid2z_mean = nn.Linear(n_hidden_r, self.z_dim)
        self.hid2z_log_sigma_sq = nn.Linear(n_hidden_r, self.z_dim)

    def _generator_netowrk(self, n_hidden_g):
        self.z2hid = nn.Linear(self.z_dim, n_hidden_g)
        self.hid2x = nn.Linear(n_hidden_g, self.input_size)

    def _sample_gaussian(self, mean, logvar):
        sample = mean + torch.exp(0.5 * logvar) * \
            logvar.new_empty(logvar.size()).normal_()
        return sample

    def forward(self, x):
        r_hid = F.relu(self.x2hid(x))
        z_mean = self.hid2z_mean(r_hid)
        z_log_sigma_sq = self.hid2z_log_sigma_sq(r_hid)
        z = self._sample_gaussian(z_mean, z_log_sigma_sq)

        g_hid = F.relu(self.z2hid(z))
        x_g = torch.sigmoid(self.hid2x(g_hid))
        return x_g, z_mean, z_log_sigma_sq

    def reconstruct(self, x):
        x = x.squeeze()
        batch_size, height, width = x.shape
        x = x.reshape(batch_size, -1)
        x_g, _, _ = self.forward(x)
        x_g = x_g.reshape(batch_size, height, width)
        return x_g

    def transform(self, x):
        x = x.squeeze()
        batch_size, height, width = x.shape
        x = x.reshape(batch_size, -1)
        _, z_mu, _ = self.forward(x)
        return z_mu


    def generate(self, z=None):
        if z is None:
            z = self._sample_gaussian(torch.zeros((1, self.z_dim)).to(self.device), torch.zeros((1, self.z_dim)).to(self.device))
        g_hid = F.relu(self.z2hid(z))
        x_g = torch.sigmoid(self.hid2x(g_hid))
        return x_g



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        batch_size = data.shape[0]
        data = data.squeeze().reshape(batch_size, -1)
        output, z_mean, z_log_sigma_sq = model(data)

        recon_term = data * torch.log(1e-10 + output) + (1 - data) * torch.log(1e-10 + 1 - output)
        recon_loss = -1 * recon_term.sum(axis=1).mean()
        if model.algo == 'vae':
            latent_term = (1 + z_log_sigma_sq - z_mean * z_mean - torch.exp(z_log_sigma_sq))
            latent_loss = -0.5 * latent_term.sum(axis=1).mean()
        loss = recon_loss + latent_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            data = data.squeeze().reshape(batch_size, -1)

            output, z_mean, z_log_sigma_sq = model(data)
            recon_term = data * torch.log(1e-10 + output) + (1 - data) * torch.log(1e-10 + 1 - output)
            recon_loss = -1 * recon_term.sum(axis=1).sum()
            if model.algo == 'vae':
                latent_term = (1 + z_log_sigma_sq - z_mean * z_mean - torch.exp(z_log_sigma_sq))
                latent_loss = -0.5 * latent_term.sum(axis=1).sum()
            loss = recon_loss + latent_loss

            test_loss += loss.item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    test_loader2 = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=100, shuffle=True, **kwargs)

    test_loader3 = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=5000, shuffle=True, **kwargs)



    model = NeuralNetwork(device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    reconstructued_images_exp(test_loader2, model, device)
    plot_latent_space_scatterplot(test_loader3, model, device)


def reconstructued_images_exp(test_loader, model, device):
    for data, target in test_loader:
        plot_mnist(data.squeeze())
        data, target = data.to(device), target.to(device)
        reconstruction = model.reconstruct(data).cpu().detach().numpy()
        plot_mnist(reconstruction)
        break

def plot_latent_space_scatterplot(test_loader, model, device):
    # Scatter plot of the lagent representations of X, colored by labels.
    for x_sample, y_sample in test_loader:
        x_sample = x_sample.to(device)
        z_mu = model.transform(x_sample).cpu().detach().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y_sample)
        plt.colorbar()
        plt.grid()
        plt.show()

        z0_min, z1_min = np.min(z_mu, axis=0)
        z0_max, z1_max = np.max(z_mu, axis=0)
        Z = [[z0, z1] for z0 in np.linspace(z0_min, z0_max, 15)
                      for z1 in np.linspace(z1_min, z1_max, 15)]
        _, height, width = x_sample.squeeze().shape
        x_reconstr = model.generate()
        x_reconstr = model.generate(torch.tensor(np.array(Z)).float().to(device))
        x_reconstr = x_reconstr.reshape(-1, height, width).cpu().detach().numpy()
        plot_mnist(x_reconstr, nrows=15, ncols=15)
        break



if __name__ == '__main__':
    main()
