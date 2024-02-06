import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image, target
        target = target.resize(image.size)
        return image, target


class ResizedCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = target.resize(image.size)

        image = F.center_crop(image, self.size)
        if target is not None:
            target = target.center_crop()

        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, target
        return image, target


class Resize2NearestPatch(object):
    """
    Customized Resize.
    Resize the short side to 224.
    Resize the long side to the nearest multiple of patch_size.
    """
    def __init__(self, scale=224, patch_size=16):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, tensor, target=None):
        H, W = tensor.shape[-2:]
        if H > W:
            l, s = H, W
            new_l = int(np.round(l/s * self.scale / self.patch_size)) * self.patch_size
            tensor_res = torch.nn.functional.interpolate(tensor[None], size=(new_l, self.scale), mode="bilinear")[0]
            return tensor_res, target.resize((self.scale, new_l))
        else:   # H < W
            s, l = H, W
            new_l = int(np.round(l/s * self.scale / self.patch_size)) * self.patch_size
            tensor_res = torch.nn.functional.interpolate(tensor[None], size=(self.scale, new_l), mode="bilinear")[0]
            return tensor_res, target.resize((new_l, self.scale))