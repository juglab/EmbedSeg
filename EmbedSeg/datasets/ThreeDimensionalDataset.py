import glob
import os
import random
import numpy as np
import tifffile
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset

class ThreeDimensionalDataset(Dataset):
    """
        ThreeDimensionalDataset class
        This class is not functional currently for one-hot instances!
    """
    def __init__(self, data_dir='./', center='center-medoid', type="train", bg_id=0, size=None, transform=None,
                 one_hot=False):
        print('3-D `{}` dataloader created! Accessing data from {}/{}/'.format(type, data_dir, type))

        # get image and instance list
        image_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images/*.tif'))
        image_list.sort()
        print('Number of images in `{}` directory is {}'.format(type, len(image_list)))
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), 'masks/*.tif'))
        instance_list.sort()
        print('Number of instances in `{}` directory is {}'.format(type, len(instance_list)))
        self.instance_list = instance_list

        center_image_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), center + '/*.tif'))
        center_image_list.sort()
        print('Number of center images in `{}` directory is {}'.format(type, len(center_image_list)))
        print('*************************')
        self.center_image_list = center_image_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.one_hot = one_hot

    def convert_zyx_to_czyx(self, im, key):
        im = im[np.newaxis, ...] # CZYX
        return im


    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        image = tifffile.imread(self.image_list[index])  # ZYX
        image = self.convert_zyx_to_czyx(image, key='image') # CZYX
        sample['image'] = image  # CZYX
        sample['im_name'] = self.image_list[index]
        if (len(self.instance_list) != 0):
            instance = tifffile.imread(self.instance_list[index])  # ZYX
            instance, label = self.decode_instance(instance, self.one_hot, self.bg_id)
            instance = self.convert_zyx_to_czyx(instance, key='instance')  # CZYX
            label = self.convert_zyx_to_czyx(label, key='label')  # CZYX
            sample['instance'] = instance
            sample['label'] = label
        if (len(self.center_image_list) != 0):
            center_image = tifffile.imread(self.center_image_list[index]) # ZYX
            center_image = self.convert_zyx_to_czyx(center_image, key='center_image')  # CZYX
            sample['center_image'] = center_image

        # transform
        if (self.transform is not None):
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, one_hot, bg_id=None):
        pic = np.array(pic, copy=False, dtype=np.uint16)
        instance_map = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.int16)
        class_map = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id

            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])

                instance_map[mask] = ids
                if (one_hot):
                    class_map[np.max(mask, axis=0)] = 1
                else:
                    class_map[mask] = 1

        return instance_map, class_map

