import collections
import numpy as np
import torch
from torchvision.transforms import transforms as T


class RandomRotationsAndFlips(T.RandomRotation):

    def __init__(self, keys=[], one_hot=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys

        self.one_hot = one_hot

    def __call__(self, sample):

        angle = self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)

        for idx, k in enumerate(self.keys):
            assert (k in sample)
            if (self.one_hot and k == 'instance'):
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (2, 3)))
            else:
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (1, 2)))
            if flip == 0:
                sample[k] = temp
            else:
                if (self.one_hot and k == 'instance'):
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=2))  # flip about D - axis
                else:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=1))  # flip about Y - axis
        return sample


class RandomRotationsAndFlips_3d(T.RandomRotation):

    def __init__(self, keys=[], one_hot = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.one_hot = one_hot


    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        times = np.random.choice(4)
        flip = np.random.choice(2)
        dir_rot = np.random.choice(3)
        dir_flip = np.random.choice(3)

        for idx, k in enumerate(self.keys):

            assert(k in sample)
            if dir_rot == 0: # rotate about ZY
                temp = np.ascontiguousarray(np.rot90(sample[k], 2*times, (1, 2)))
            elif dir_rot == 1: # rotate about YX
                temp = np.ascontiguousarray(np.rot90(sample[k], times, (2, 3)))
            elif dir_rot == 2: # rotate about ZX
                temp = np.ascontiguousarray(np.rot90(sample[k], 2*times, (3,1)))

            if flip == 0:
                sample[k] = temp
            else:
                if dir_flip == 0:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=1)) # Z
                elif dir_flip == 1:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=2)) # Y
                elif dir_flip == 2:
                    sample[k] = np.ascontiguousarray(np.flip(temp, axis=3))  # X

        return sample




class ToTensorFromNumpy(object):
    def __init__(self, keys=[], type="float", normalization_factor=255):

        if isinstance(type, collections.Iterable):
            assert (len(keys) == len(type))

        self.keys = keys
        self.type = type
        self.normalization_factor = normalization_factor

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):
            #assert (k in sample)

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]
            if (k in sample):
                if k == 'image':  # image
                    sample[k] = torch.from_numpy(sample[k].astype("float32")).float().div(self.normalization_factor)
                elif k =='instance' or k=='label' or k=='center-image':
                    sample[k] = torch.from_numpy(sample[k]).short()
        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
