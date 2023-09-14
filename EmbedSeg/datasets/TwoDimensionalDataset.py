import ast
import glob
import numpy as np
import os
import pandas as pd
import pycocotools.mask as rletools
import random
import tifffile
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset

from EmbedSeg.utils.generate_crops import (
    normalize_min_max_percentile,
    normalize_mean_std,
)


class TwoDimensionalDataset(Dataset):
    """
    A class used to create a PyTorch Dataset for handling 2D images and label masks

    Attributes
    ----------
    image_list: list of strings containing paths to the images

    instance_list: list of strings containing paths to the GT instances

    center_image_list: list of strings containing paths to the center images

    bg_id: int
        Id corresponding to the background, default=0
    size: int
        This is set equal to a different value than `real_size`
        in case the epoch is desired to be shorter

    real_size: int
        Actual number of the number of images

    transform: PyTorch transform

    one_hot: bool
        Should be set equal to True

    norm: str
        Should be set equal to one of `min-max-percentile`, `absolute`, `mean-std`

    type: str
        Should be set equal to one of `train`, `val` or `test`
    data_type: str
        Should be set equal to one of `8-bit` or `16-bit`

    normalization: bool
        Should be equal to True for test images and set equal to False
        during training (since crops are already normalized)

    Methods
    -------
    __init__: Initializes an object of class `TwoDimensionalDataset`

    __len__: Returns self.real_size if self.size = None

    convert_yx_to_cyx: Adds an additional dimension for channel

    decode_instance: In case, instances are available as tiffs,
    this method decodes them

    rle_decode: In case, instances are available as csv files (and not tiffs),
    this method decodes them
    """

    def __init__(
        self,
        data_dir="./",
        center="center-medoid",
        type="train",
        bg_id=0,
        size=None,
        transform=None,
        one_hot=False,
        norm="min-max-percentile",
        data_type="8-bit",
        normalization=False,
        anisotropy_factor=1.0,
        sliced_mode=False,
        uniform_ds_factor=1,
    ):
        print(
            "2-D `{}` dataloader created! Accessing data from {}/{}/".format(
                type, data_dir, type
            )
        )

        # get image and instance list
        image_list = glob.glob(
            os.path.join(data_dir, "{}/".format(type), "images/*.tif")
        )
        image_list.sort()
        print("Number of images in `{}` directory is {}".format(type, len(image_list)))
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(data_dir, "{}/".format(type), "masks/*"))
        instance_list.sort()
        print(
            "Number of instances in `{}` directory is {}".format(
                type, len(instance_list)
            )
        )
        self.instance_list = instance_list

        center_image_list = glob.glob(
            os.path.join(data_dir, "{}/".format(type), center + "/*")
        )
        center_image_list.sort()
        print(
            "Number of center images in `{}` directory is {}".format(
                type, len(center_image_list)
            )
        )
        print("*************************")
        self.center_image_list = center_image_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.one_hot = one_hot
        self.norm = norm
        self.type = type
        self.data_type = data_type
        self.normalization = normalization

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def convert_yx_to_cyx(self, im, key):
        if im.ndim == 2 and key == "image":  # gray-scale image
            im = im[np.newaxis, ...]  # CYX
        elif im.ndim == 3 and key == "image":  # multi-channel image image
            pass
        else:
            im = im[np.newaxis, ...]
        return im

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        image = tifffile.imread(self.image_list[index])  # YX or CYX
        if self.normalization and self.norm == "min-max-percentile":
            if image.ndim == 2:  # gray-scale
                image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))
            elif image.ndim == 3:  # multi-channel image (C, H, W)
                image = normalize_min_max_percentile(image, 1, 99.8, axis=(1, 2))
        elif self.normalization and self.norm == "mean-std":
            if image.ndim == 2:
                image = normalize_mean_std(image)  # axis == None
            elif image.ndim == 3:
                image = normalize_mean_std(image, axis=(1, 2))
        elif self.normalization and self.norm == "absolute":
            image = image.astype(float)
            if self.data_type == "8-bit":
                image /= 255
            elif self.data_type == "16-bit":
                image /= 65535
        image = self.convert_yx_to_cyx(image, key="image")
        sample["image"] = image  # CYX
        sample["im_name"] = self.image_list[index]

        if len(self.instance_list) != 0:
            if self.instance_list[index][-3:] == "csv":
                instance = self.rle_decode(
                    self.instance_list[index], one_hot=self.one_hot
                )
            else:
                instance = tifffile.imread(
                    self.instance_list[index]
                )  # YX or DYX (one-hot!)
            instance, label = self.decode_instance(instance, self.one_hot, self.bg_id)
            instance = self.convert_yx_to_cyx(instance, key="instance")  # CYX or CDYX
            label = self.convert_yx_to_cyx(label, key="label")  # CYX
            sample["instance"] = instance
            sample["label"] = label
        if len(self.center_image_list) != 0:
            if self.center_image_list[index][-3:] == "csv":
                center_image = self.rle_decode(
                    self.center_image_list[index], center=True
                )
            else:
                center_image = tifffile.imread(self.center_image_list[index])
            center_image = self.convert_yx_to_cyx(
                center_image, key="center_image"
            )  # CYX
            sample["center_image"] = center_image

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, one_hot, bg_id=None):
        pic = np.array(pic, copy=False, dtype=np.uint16)
        if one_hot:
            instance_map = np.zeros(
                (pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.uint8
            )
            class_map = np.zeros((pic.shape[1], pic.shape[2]), dtype=np.uint8)
        else:
            instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.int16)
            class_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                if one_hot:
                    class_map[np.max(mask, axis=0)] = 1
                else:
                    class_map[mask] = 1

        return instance_map, class_map

    @classmethod
    def rle_decode(cls, filename, one_hot=False, center=False):
        df = pd.read_csv(filename, header=None)
        df_numpy = df.to_numpy()
        d = {}

        if one_hot:
            mask_decoded = []
            for row in df_numpy:
                d["counts"] = ast.literal_eval(row[1])
                d["size"] = ast.literal_eval(row[2])
                mask = rletools.decode(d)  # returns binary mask
                mask_decoded.append(mask)
        else:
            if center:
                mask_decoded = np.zeros(
                    ast.literal_eval(df_numpy[0][2]), dtype=np.bool
                )  # obtain size by reading first row of csv file
            else:
                mask_decoded = np.zeros(
                    ast.literal_eval(df_numpy[0][2]), dtype=np.uint16
                )  # obtain size by reading first row of csv file
            for row in df_numpy:
                d["counts"] = ast.literal_eval(row[1])
                d["size"] = ast.literal_eval(row[2])
                mask = rletools.decode(d)  # returns binary mask
                y, x = np.where(mask == 1)
                mask_decoded[y, x] = int(row[0])
        return np.asarray(mask_decoded)
