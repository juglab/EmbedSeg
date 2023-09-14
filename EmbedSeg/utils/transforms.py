import collections

import numpy as np
import torch
from torchvision.transforms import transforms as T


class RandomRotationsAndFlips(T.RandomRotation):
    """
    A class used to represent Random Rotations and Flips for Augmenting
    2D Image Data

    ...

    Attributes
    ----------
    keys : dictionary
        keys include `instance`, `label`, `center-image`
        See `TwoDimensionalDataset.py`
    one_hot : bool
        Should be set to True, if the GT label masks are present
        in a one-hot encoded fashion

    Methods
    -------
    __call__: Returns rotated or flipped image, instance label mask and
    center image

    """

    def __init__(self, keys=[], one_hot=False, *args, **kwargs):
        """
        Parameters
        ----------
        keys : dictionary
            keys include `instance`, `label`, `center-image`
            See `TwoDimensionalDataset.py`
        one_hot : bool
            Should be set to True, if the GT label masks are present in a
            one-hot encoded fashion

        """

        super().__init__(*args, **kwargs)
        self.keys = keys
        self.one_hot = one_hot

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample

        Returns
        ----------
        sample

        """

        self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)

        for idx, k in enumerate(self.keys):
            assert k in sample
            if self.one_hot and k == "instance":
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (2, 3)))
            else:
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
            if flip == 0:
                sample[k] = temp
            else:
                if self.one_hot and k == "instance":
                    sample[k] = np.ascontiguousarray(
                        np.flip(temp, axis=2)
                    )  # flip about D - axis
                else:
                    sample[k] = np.ascontiguousarray(
                        np.flip(temp, axis=1)
                    )  # flip about Y - axis
        return sample


class RandomRotationsAndFlips_3d(T.RandomRotation):
    """
    A class used to represent Random Rotations and Flips for Augmenting
    3D Image Data

    ...

    Attributes
    ----------
    keys : dictionary
        keys include `instance`, `label`, `center-image`
        See `ThreeDimensionalDataset.py`
    one_hot : bool
        Should be set to True, if the GT label masks are present in a one-hot
        encoded fashion
        Not applicable to 3D. This parameter will be deprecated in a future release

    Methods
    -------
    __call__: Returns rotated or flipped image, instance label mask and
    center image

    """

    def __init__(self, keys=[], one_hot=False, *args, **kwargs):
        """
        Parameters
        ----------
        keys : dictionary
            keys include `instance`, `label`, `center-image`
            See `ThreeDimensionalDataset.py`
        one_hot : bool
            Should be set to True, if the GT label masks are present
            in a one-hot encoded fashion
            Not applicable to 3D. This parameter will be deprecated
            in a future release

        """

        super().__init__(*args, **kwargs)
        self.keys = keys
        self.one_hot = one_hot

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample

        Returns
        ----------
        sample

        """
        self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)
        dir_rot = np.random.choice(3)
        dir_flip = np.random.choice(3)

        for idx, k in enumerate(self.keys):
            assert k in sample
            if dir_rot == 0:  # rotate about ZY
                temp = np.ascontiguousarray(np.rot90(sample[k], 2 * times, (1, 2)))
            elif dir_rot == 1:  # rotate about YX
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (2, 3)))
            elif dir_rot == 2:  # rotate about ZX
                temp = np.ascontiguousarray(np.rot90(sample[k], 2 * times, (3, 1)))

            if flip == 0:
                sample[k] = temp
            else:
                if dir_flip == 0:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=1))  # Z
                elif dir_flip == 1:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=2))  # Y
                elif dir_flip == 2:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=3))  # X

        return sample


class ToTensorFromNumpy(object):
    """
    A class used to convert numpy arrays to PyTorch tensors

    ...

    Attributes
    ----------
    keys : dictionary
        keys include `instance`, `label`, `center-image`, `image`
    type : str

    normalization_factor: float

    Methods
    -------
    __call__: Returns Pytorch Tensors

    """

    def __init__(self, keys=[], type="float", normalization_factor=1.0):
        if isinstance(type, collections.abc.Iterable):
            assert len(keys) == len(type)

        self.keys = keys
        self.type = type
        self.normalization_factor = normalization_factor

    def __call__(self, sample):
        for idx, k in enumerate(self.keys):
            # assert (k in sample)

            t = self.type
            if isinstance(t, collections.abc.Iterable):
                t = t[idx]
            if k in sample:
                if k == "image":  # image
                    sample[k] = (
                        torch.from_numpy(sample[k].astype("float32"))
                        .float()
                        .div(self.normalization_factor)
                    )
                elif k == "instance" or k == "label":
                    sample[k] = torch.from_numpy(
                        sample[k]
                    )  # np.int16 to torch.int16 or short
                elif k == "center-image":
                    sample[k] = torch.from_numpy(sample[k])  # np.bool to torch.Bool
        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr["name"]
        opts = tr["opts"]

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
