import collections
import os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


class ShepardMetzler(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Image.open(io.BytesIO(x)))

        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])

        viewpoints = torch.from_numpy(data.cameras)
        viewpoints = viewpoints.view(-1, 5)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints