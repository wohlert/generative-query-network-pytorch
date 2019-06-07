import torch
from torch.utils.data import Dataset


class PlaceholderData(Dataset):
    """
    Random placeholder dataset for testing
    training loop without loading actual data.
    """
    def __init__(self, *args, **kwargs):
        super(PlaceholderData, self).__init__()

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        # (b, m, c, h, w)
        images = torch.randn(64, 15, 3, 64, 64)

        # (b, m, 5)
        viewpoints = torch.randn(64, 15, 7)

        return images, viewpoints
