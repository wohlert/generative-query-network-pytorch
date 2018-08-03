import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


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