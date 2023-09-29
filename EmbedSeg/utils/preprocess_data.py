import os
import shutil
import subprocess as sp
import urllib.request
import zipfile
from glob import glob

import numpy as np
import tifffile
from tqdm import tqdm


def extract_data(zip_url, project_name, data_dir="../../../data/"):
    """
    Extracts data from `zip_url` to the location identified by
    `data_dir` and `project_name` parameters.

    Parameters
    ----------
    zip_url: string
        Indicates the external url
    project_name: string
        Indicates the path to the sub-directory at the location
        identified by the parameter `data_dir`
    data_dir: string
        Indicates the path to the directory where the data should be saved.
    Returns
    -------

    """
    zip_path = os.path.join(data_dir, project_name + ".zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created new directory {}".format(data_dir))

    if os.path.exists(zip_path):
        print("Zip file was downloaded and extracted before!")
    else:
        if os.path.exists(os.path.join(data_dir, project_name, "download/")):
            pass
        else:
            os.makedirs(os.path.join(data_dir, project_name, "download/"))
            urllib.request.urlretrieve(zip_url, zip_path)
            print("Downloaded data as {}".format(zip_path))
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            if os.path.exists(
                os.path.join(data_dir, os.path.basename(zip_url)[:-4], "train")
            ):
                shutil.move(
                    os.path.join(data_dir, os.path.basename(zip_url)[:-4], "train"),
                    os.path.join(data_dir, project_name, "download/"),
                )
            if os.path.exists(
                os.path.join(data_dir, os.path.basename(zip_url)[:-4], "val")
            ):
                shutil.move(
                    os.path.join(data_dir, os.path.basename(zip_url)[:-4], "val"),
                    os.path.join(data_dir, project_name, "download/"),
                )
            if os.path.exists(
                os.path.join(data_dir, os.path.basename(zip_url)[:-4], "test")
            ):
                shutil.move(
                    os.path.join(data_dir, os.path.basename(zip_url)[:-4], "test"),
                    os.path.join(data_dir, project_name, "download/"),
                )
            print(
                "Unzipped data to {}".format(
                    os.path.join(data_dir, project_name, "download/")
                )
            )


def make_dirs(data_dir, project_name):
    """
    Makes directories - `train`, `val, `test` and subdirectories
    under each `images` and `masks`

    Parameters
    ----------
    data_dir: string
        Indicates the path where the `project` lives.
    project_name: string
        Indicates the name of the sub-folder under the location
        identified by `data_dir`.

    Returns
    -------
    """
    image_path_train = os.path.join(data_dir, project_name, "train", "images/")
    instance_path_train = os.path.join(data_dir, project_name, "train", "masks/")
    image_path_val = os.path.join(data_dir, project_name, "val", "images/")
    instance_path_val = os.path.join(data_dir, project_name, "val", "masks/")
    image_path_test = os.path.join(data_dir, project_name, "test", "images/")
    instance_path_test = os.path.join(data_dir, project_name, "test", "masks/")

    if not os.path.exists(image_path_train):
        os.makedirs(os.path.dirname(image_path_train))
        print("Created new directory : {}".format(image_path_train))

    if not os.path.exists(instance_path_train):
        os.makedirs(os.path.dirname(instance_path_train))
        print("Created new directory : {}".format(instance_path_train))

    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))

    if not os.path.exists(image_path_test):
        os.makedirs(os.path.dirname(image_path_test))
        print("Created new directory : {}".format(image_path_test))

    if not os.path.exists(instance_path_test):
        os.makedirs(os.path.dirname(instance_path_test))
        print("Created new directory : {}".format(instance_path_test))


def split_train_crops(
    project_name,
    center,
    crops_dir="crops",
    subset=0.15,
    by_fraction=True,
    train_name="train",
    seed=1000,
):
    """
    Moves a fraction of crops (specified by `subset`) extracted from
    train images into a separate directory

    Parameters
    ----------
    project_name: string
           Name of dataset. for example, set this to 'dsb-2018', or 'bbbc010-2012'
    center: string
            Set this to 'medoid' or 'centroid'
    crops_dir: string
            Name of the crops directory. Default value = 'crops'
    subset: int/float
            if by_fraction is True, then subset should be set equal to the
            percentage of image crops reserved for validation.
            if by_fraction is False, then subset should be set equal to the
            number of image crops reserved for validation.
    by_fraction: boolean
            if True, then reserve a fraction <1 of image crops for validation
            if False, then reserve the absolute number of image crops
            as specified by `subset` for validation
    train_name: string
            name of directory containing train image and instance crops
    seed: int

    Returns
    -------
    """

    image_dir = os.path.join(crops_dir, project_name, train_name, "images")
    instance_dir = os.path.join(crops_dir, project_name, train_name, "masks")
    center_dir = os.path.join(crops_dir, project_name, train_name, "center-" + center)

    image_names = sorted(glob(os.path.join(image_dir, "*.tif")))
    instance_names = sorted(
        glob(os.path.join(instance_dir, "*"))
    )  # this could be `tifs` or `csvs`
    center_names = sorted(
        glob(os.path.join(center_dir, "*"))
    )  # this could be `tifs` or `csvs`

    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if by_fraction:
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)

    valIndices = indices[:subset_len]

    image_path_val = os.path.join(crops_dir, project_name, "val", "images/")
    instance_path_val = os.path.join(crops_dir, project_name, "val", "masks/")
    center_path_val = os.path.join(
        crops_dir, project_name, "val", "center-" + center + "/"
    )

    val_images_exist = False
    val_masks_exist = False
    val_center_images_exist = False

    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))
    else:
        val_images_exist = True

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))
    else:
        val_masks_exist = True

    if not os.path.exists(center_path_val):
        os.makedirs(os.path.dirname(center_path_val))
        print("Created new directory : {}".format(center_path_val))
    else:
        val_center_images_exist = True

    if not val_images_exist and not val_masks_exist and not val_center_images_exist:
        for val_index in valIndices:
            shutil.move(
                image_names[val_index],
                os.path.join(crops_dir, project_name, "val", "images"),
            )
            shutil.move(
                instance_names[val_index],
                os.path.join(crops_dir, project_name, "val", "masks"),
            )
            shutil.move(
                center_names[val_index],
                os.path.join(crops_dir, project_name, "val", "center-" + center),
            )

        print(
            "Val Images/Masks/Center-{}-image crops saved at {}".format(
                center, os.path.join(crops_dir, project_name, "val")
            )
        )
    else:
        print(
            "Val Images/Masks/Center-{}-image crops already available at {}".format(
                center, os.path.join(crops_dir, project_name, "val")
            )
        )


def split_train_val(
    data_dir, project_name, train_val_name, subset=0.15, by_fraction=True, seed=1000
):
    """
    Splits the `train` directory into `val` directory
    using the partition percentage of `subset`.

    Parameters
    ----------
    data_dir: string
        Indicates the path where the `project` lives.
    project_name: string
        Indicates the name of the sub-folder under the location
        identified by `data_dir`.
    train_val_name: string
        Indicates the name of the sub-directory under `project-name`
        which must be split
    subset: float
        Indicates the fraction of data to be reserved for validation
    seed: integer
        Allows for the same partition to be used in each experiment.
        Change this if you would like to obtain results with different
        train-val partitions.
    Returns
    -------

    """

    image_dir = os.path.join(
        data_dir, project_name, "download", train_val_name, "images"
    )
    instance_dir = os.path.join(
        data_dir, project_name, "download", train_val_name, "masks"
    )
    image_names = sorted(glob(os.path.join(image_dir, "*.tif")))
    instance_names = sorted(glob(os.path.join(instance_dir, "*.tif")))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if by_fraction:
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)
    val_indices = indices[:subset_len]
    trainIndices = indices[subset_len:]
    make_dirs(data_dir=data_dir, project_name=project_name)

    for val_index in val_indices:
        shutil.copy(
            image_names[val_index],
            os.path.join(data_dir, project_name, "val", "images"),
        )
        shutil.copy(
            instance_names[val_index],
            os.path.join(data_dir, project_name, "val", "masks"),
        )

    for trainIndex in trainIndices:
        shutil.copy(
            image_names[trainIndex],
            os.path.join(data_dir, project_name, "train", "images"),
        )
        shutil.copy(
            instance_names[trainIndex],
            os.path.join(data_dir, project_name, "train", "masks"),
        )

    image_dir = os.path.join(data_dir, project_name, "download", "test", "images")
    instance_dir = os.path.join(data_dir, project_name, "download", "test", "masks")
    image_names = sorted(glob(os.path.join(image_dir, "*.tif")))
    instance_names = sorted(glob(os.path.join(instance_dir, "*.tif")))
    test_indices = np.arange(len(image_names))
    for test_index in test_indices:
        shutil.copy(
            image_names[test_index],
            os.path.join(data_dir, project_name, "test", "images"),
        )
        shutil.copy(
            instance_names[test_index],
            os.path.join(data_dir, project_name, "test", "masks"),
        )
    print(
        "Train-Val-Test Images/Masks copied to {}".format(
            os.path.join(data_dir, project_name)
        )
    )


def split_train_test(
    data_dir, project_name, train_test_name, subset=0.5, by_fraction=True, seed=1000
):
    """
    Splits the `train` directory into `test` directory
    using the partition percentage of `subset`.

    Parameters
    ----------
    data_dir: string
        Indicates the path where the `project` lives.
    project_name: string
        Indicates the name of the sub-folder under the location
        identified by `data_dir`.
    train_test_name: string
        Indicates the name of the sub-directory under `project-name`
        which must be split
    subset: float
        Indicates the fraction of data to be reserved for evaluation
    seed: integer
        Allows for the same partition to be used
        in each experiment.
        Change this if you would like to obtain results
        with different train-test partitions.

    Returns
    -------
    """

    image_dir = os.path.join(
        data_dir, project_name, "download", train_test_name, "images"
    )
    instance_dir = os.path.join(
        data_dir, project_name, "download", train_test_name, "masks"
    )
    image_names = sorted(glob(os.path.join(image_dir, "*.tif")))
    instance_names = sorted(glob(os.path.join(instance_dir, "*.tif")))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if by_fraction:
        subset_len = int(subset * len(image_names))
    else:
        subset_len = int(subset)
    test_indices = indices[:subset_len]
    # make_dirs(data_dir=data_dir, project_name=project_name)
    test_images_exist = False
    test_masks_exist = False
    if not os.path.exists(
        os.path.join(data_dir, project_name, "download", "test", "images")
    ):
        os.makedirs(os.path.join(data_dir, project_name, "download", "test", "images"))
        print(
            "Created new directory : {}".format(
                os.path.join(data_dir, project_name, "download", "test", "images")
            )
        )
    else:
        test_images_exist = True
    if not os.path.exists(
        os.path.join(data_dir, project_name, "download", "test", "masks")
    ):
        os.makedirs(os.path.join(data_dir, project_name, "download", "test", "masks"))
        print(
            "Created new directory : {}".format(
                os.path.join(data_dir, project_name, "download", "test", "masks")
            )
        )
    else:
        test_masks_exist = True
    if not test_images_exist and not test_masks_exist:
        for test_index in test_indices:
            shutil.move(
                image_names[test_index],
                os.path.join(data_dir, project_name, "download", "test", "images"),
            )
            shutil.move(
                instance_names[test_index],
                os.path.join(data_dir, project_name, "download", "test", "masks"),
            )
        print(
            "Train-Test Images/Masks saved at {}".format(
                os.path.join(data_dir, project_name, "download")
            )
        )
    else:
        print(
            "Train-Test Images/Masks already available at {}".format(
                os.path.join(data_dir, project_name, "download")
            )
        )


def calculate_foreground_weight(
    data_dir, project_name, train_val_name, mode, background_id=0
):
    """

    Parameters
    -------

    data_dir: string
        Name of directory containing data
    project_name: string
        Name of directory containing images and instances
    train_val_name: string
        one of 'train' or 'val'
    mode: string
        one of '2d' or '3d'
    background_id: int, default
        Id which corresponds to the background.

    Returns
    -------
    float:
        Ratio of the number of foreground pixels to the background pixels,
        averaged over all available label masks

    """
    instance_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, "masks")
        instance_names += sorted(glob(os.path.join(instance_dir, "*.tif")))

    statistics = []
    for i in tqdm(range(len(instance_names)), position=0, leave=True):
        ma = tifffile.imread(instance_names[i])
        if mode in ["2d", "3d_sliced", "3d_ilp"]:
            statistics.append(10.0)
        elif mode == "3d":
            z, y, x = np.where(ma == background_id)
            len_bg = len(z)
            z, y, x = np.where(ma > background_id)
            len_fg = len(z)
            statistics.append(len_bg / len_fg)
    print(
        "Foreground weight of the `{}` dataset set equal to {:.3f}".format(
            project_name, np.mean(statistics)
        )
    )
    return np.mean(statistics)


def calculate_object_size(
    data_dir, project_name, train_val_name, mode, one_hot, process_k, background_id=0
):
    """
    Calculate the mean object size from the available label masks

    Parameters
    -------

    data_dir: string
        Name of directory storing the data. For example, 'data'
    project_name: string
        Name of directory containing data specific to this project.
        For example, 'dsb-2018'
    train_val_name: string
        Name of directory containing 'train' and 'val' images and instance masks
    mode: string
        One of '2d', '3d', '3d_sliced', '3d_ilp'
    one_hot: boolean
        set to True, if instances are encoded as one-hot
    process_k: tuple (int, int)
        Parameter for speeding up the calculation of the object size
        by considering only a fewer number of images and objects
        The first argument in the tuple is the number of images one should consider
        The second argument in the tuple is the number of objects in the image,
        one should consider
    background_id: int
        Id which corresponds to the background.

    Returns
    -------
    (float, float, float, float, float, float, float, float, float)
    (minimum number of pixels in an object,
     mean number of pixels in an object,
     max number of pixels in an object,
    mean number of pixels along the x dimension,
     standard deviation of number of pixels along the x dimension,
    mean number of pixels along the y dimension,
     standard deviation of number of pixels along the y dimension,
    mean number of pixels along the z dimension,
     standard deviation of number of pixels along the z dimension)

    """

    instance_names = []
    size_list_x = []
    size_list_y = []
    size_list_z = []
    size_list = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, "masks")
        instance_names += sorted(glob(os.path.join(instance_dir, "*.tif")))

    if process_k is not None:
        n_images = process_k[0]
    else:
        n_images = len((instance_names))
    for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
        ma = tifffile.imread(instance_names[i])
        if one_hot and mode == "2d":
            for z in range(ma.shape[0]):
                y, x = np.where(ma[z] == 1)
                size_list_x.append(np.max(x) - np.min(x))
                size_list_y.append(np.max(y) - np.min(y))
                size_list.append(len(x))
        elif not one_hot and mode == "2d":
            ids = np.unique(ma)
            ids = ids[ids != background_id]
            for id in ids:
                y, x = np.where(ma == id)
                size_list_x.append(np.max(x) - np.min(x))
                size_list_y.append(np.max(y) - np.min(y))
                size_list.append(len(x))
        elif not one_hot and mode in ["3d", "3d_sliced"]:
            ids = np.unique(ma)
            ids = ids[ids != background_id]
            if process_k is not None:
                n_ids = process_k[1]
            else:
                n_ids = len(ids)
            for id in tqdm(ids[:n_ids], position=0, leave=True):
                # for id in ids:
                z, y, x = np.where(ma == id)
                size_list_z.append(np.max(z) - np.min(z))
                size_list_y.append(np.max(y) - np.min(y))
                size_list_x.append(np.max(x) - np.min(x))
                size_list.append(len(x))
        elif not one_hot and mode == "3d_ilp":
            for z in range(ma.shape[0]):
                ids = np.unique(ma[z])
                ids = ids[ids != background_id]
                for id in ids:
                    y, x = np.where(ma[z] == id)
                    size_list_y.append(np.max(y) - np.min(y))
                    size_list_x.append(np.max(x) - np.min(x))
                    size_list.append(len(x))

    print(
        "Minimum object size of the `{}` dataset is equal to {}".format(
            project_name, np.min(size_list)
        )
    )
    print(
        "Mean object size of the `{}` dataset is equal to {}".format(
            project_name, np.mean(size_list)
        )
    )
    print(
        "Maximum object size of the `{}` dataset is equal to {}".format(
            project_name, np.max(size_list)
        )
    )
    print(
        "Average object size of the `{}` dataset along `x` = {:.3f}".format(
            project_name, np.mean(size_list_x)
        )
    )
    print(
        "Std. dev object size of the `{}` dataset along `x` = {:.3f}".format(
            project_name, np.std(size_list_x)
        )
    )
    print(
        "Average object size of the `{}` dataset along `y` = {:.3f}".format(
            project_name, np.mean(size_list_y)
        )
    )
    print(
        "Std. dev object size of the `{}` dataset along `y` = {:.3f}".format(
            project_name, np.std(size_list_y)
        )
    )

    if mode in ["3d", "3d_sliced"]:
        print(
            "Average object size of the `{}` dataset along `z` = {:.3f}".format(
                project_name, np.mean(size_list_z)
            )
        )
        print(
            "Std. dev object size of the `{}` dataset along `z` =  {:.3f}".format(
                project_name, np.std(size_list_z)
            )
        )
        return (
            np.min(size_list).astype(float),
            np.mean(size_list).astype(float),
            np.max(size_list).astype(float),
            np.mean(size_list_z).astype(float),
            np.mean(size_list_y).astype(float),
            np.mean(size_list_x).astype(float),
            np.std(size_list_z).astype(float),
            np.std(size_list_y).astype(float),
            np.std(size_list_x).astype(float),
        )

    else:
        return (
            np.min(size_list).astype(float),
            np.mean(size_list).astype(float),
            np.max(size_list).astype(float),
            None,
            np.mean(size_list_y).astype(float),
            np.mean(size_list_x).astype(float),
            None,
            np.std(size_list_y).astype(float),
            np.std(size_list_x).astype(float),
        )


def get_gpu_memory():
    """
    Identifies the max memory on the operating GPU
    https://stackoverflow.com/questions/59567226/how-to-programmatically
    -determine-available-gpu-memory-with-tensorflow/59571639#59571639

    """

    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    return memory_total_values


def round_up_8(x):
    """
    Rounds up `x` to the next nearest multiple of 8
    for e.g. round_up_8(7) = 8

    Parameters
    -------

    x: int

    Returns
    -------
    int
    Next nearest multiple to 8

    """
    return (x.astype(int) + 7) & (-8)


def calculate_max_eval_image_size(
    data_dir, project_name, test_name, mode, anisotropy_factor=1.0, scale_factor=4.0
):
    """
    Identifies the tile size to be used during training and evaluation.
    We look for the largest evaluation image.
    If the entire image could fit on the available GPU memory
    (based on an empirical idea), then  the tile size is
    set equal to those dimensions.
    If the dimensions are larger, then a smaller tile size
    as specified by the GPU memory is used.

    Parameters
    -------

    data_dir: string
        Name of directory storing the data. For example, 'data'
    project_name: string
        Name of directory containing data specific to this project.
        For example, 'dsb-2018'
    train_val_name: string
        Name of directory containing 'train' and 'val' images and instance masks
    mode: string
        One of '2d', '3d', '3d_sliced', '3d_ilp'
    anisotropy_factor: float
        Ratio of the pixel size along the z dimension to the pixel size
        along the x/y dimension
        Here we assume that the pixel size along the x and y dimension is the same
    scale_factor: float, default
        Used to evaluate the maximum GPU memory which
        shall be used during evaluation

    Returns
    -------

    (int, int, int)
    (tile size along z, tile size along y, tile size along x)

    """

    image_names = []
    size_z_list = []
    size_y_list = []
    size_x_list = []
    for name in test_name:
        instance_dir = os.path.join(data_dir, project_name, name, "images")
        image_names += sorted(glob(os.path.join(instance_dir, "*.tif")))

    for i in tqdm(range(len(image_names)), position=0, leave=True):
        im = tifffile.imread(image_names[i])
        if mode == "2d":
            size_y_list.append(im.shape[0])
            size_x_list.append(im.shape[1])
        elif mode in ["3d", "3d_sliced", "3d_ilp"]:
            size_z_list.append(im.shape[0])
            size_y_list.append(im.shape[1])
            size_x_list.append(im.shape[2])
    if mode in ["2d", "3d_ilp", "3d_sliced"]:
        max_y = np.max(size_y_list)
        max_x = np.max(size_x_list)
        max_y, max_x = round_up_8(max_y), round_up_8(max_x)
        max_x_y = np.maximum(max_x, max_y)

        max_x, max_y = max_x_y, max_x_y
        print(
            "Tile size of the `{}` dataset set equal to ({}, {})".format(
                project_name, max_y, max_x
            )
        )
        if mode == "3d_sliced":
            return max_y.astype(float), max_y.astype(float), max_x.astype(float)
        else:
            return None, max_y.astype(float), max_x.astype(float)
    elif mode == "3d":
        max_z = np.max(size_z_list)
        max_y = np.max(size_y_list)
        max_x = np.max(size_x_list)

        max_z = round_up_8(max_z)
        max_y = round_up_8(max_y)
        max_x = round_up_8(max_x)

        max_x_y = np.maximum(max_x, max_y)

        max_x = max_x_y
        max_y = max_x_y
        max_z = max_z

        print(
            "Tile size of the `{}` dataset set equal to  \
                    (n_z = {}, n_y = {}, n_x = {})".format(
                project_name, max_z, max_y, max_x
            )
        )
        return max_z.astype(float), max_y.astype(float), max_x.astype(float)


def calculate_avg_background_intensity(
    data_dir, project_name, train_val_name, one_hot, background_id=0
):
    """
    Calculates the average intensity in the regions of the raw image
    which corresponds to the background label

    Parameters
    -------

    data_dir: str
        Path to directory containing all data
    project_name: str
        Path to directory containing project-specific images and instances
    train_val_name: str
        One of 'train' or 'val'
    one_hot: str
        Set to True, if instances are encoded in a one-hot fashion
    background_id: int
         Label corresponding to the background

    Returns
    -------
        float
        Average background intensity of the dataset
    """

    instance_names = []
    image_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, "masks")
        image_dir = os.path.join(data_dir, project_name, name, "images")
        instance_names += sorted(glob(os.path.join(instance_dir, "*.tif")))
        image_names += sorted(glob(os.path.join(image_dir, "*.tif")))
    statistics = []
    if one_hot:
        for i in tqdm(range(len(instance_names)), position=0, leave=True):
            ma = tifffile.imread(instance_names[i])
            bg_mask = ma == 0
            im = tifffile.imread(image_names[i])
            statistics.append(np.average(im[np.min(bg_mask, 0)]))
    else:
        for i in tqdm(range(len(instance_names)), position=0, leave=True):
            ma = tifffile.imread(instance_names[i])
            bg_mask = ma == background_id
            im = tifffile.imread(image_names[i])
            if im.ndim == ma.ndim:
                statistics.append(np.average(im[bg_mask]))
            elif im.ndim == ma.ndim + 1:  # multi-channel image
                statistics.append(np.average(im[:, bg_mask], axis=1))
    if im.ndim == ma.ndim:
        print(
            "Average background intensity of the `{}` dataset set \
                    equal to {:.3f}".format(
                project_name, np.mean(statistics, 0)
            )
        )
    elif im.ndim == ma.ndim + 1:
        print(
            "Average background intensity of the `{}` dataset set \
                    equal to {}".format(
                project_name, np.mean(statistics, 0)
            )
        )
    return np.mean(statistics, 0).tolist()


def get_data_properties(
    data_dir,
    project_name,
    train_val_name,
    test_name,
    mode,
    one_hot=False,
    process_k=None,
    anisotropy_factor=1.0,
    background_id=0,
):
    """

    Parameters
    -------

    data_dir: string
            Path to directory containing all data
    project_name: string
            Path to directory containing project-specific images and instances
    train_val_name: string
            One of 'train' or 'val'
    test_name: string
            Name of test directory.
    mode: string
            One of '2d', '3d', '3d_sliced', '3d_ilp'
    one_hot: boolean
            set to True, if instances are encoded in a one-hot fashion
    process_k (int, int)
            first `int` argument in tuple specifies number of images
            which must be processed
            second `int` argument in tuple specifies number of ids
            which must be processed
    anisotropy_factor: float
            Ratio of the real-world size of the z-dimension to the
            x or y dimension in the raw images
            If the image is down-sampled along the z-dimension,
            then `anisotropy_factor` is greater than 1.0
    background_id: int
            Label id corresponding to the background

    Returns
    -------
    data_properties_dir: dictionary
            keys include `foreground_weight`, `min_object_size`,
            `project_name`, `avg_background_intensity` etc

    """
    data_properties_dir = {}
    data_properties_dir["foreground_weight"] = calculate_foreground_weight(
        data_dir, project_name, train_val_name, mode, background_id=background_id
    )
    (
        data_properties_dir["min_object_size"],
        data_properties_dir["mean_object_size"],
        data_properties_dir["max_object_size"],
        data_properties_dir["avg_object_size_z"],
        data_properties_dir["avg_object_size_y"],
        data_properties_dir["avg_object_size_x"],
        data_properties_dir["stdev_object_size_z"],
        data_properties_dir["stdev_object_size_y"],
        data_properties_dir["stdev_object_size_x"],
    ) = calculate_object_size(
        data_dir,
        project_name,
        train_val_name,
        mode,
        one_hot,
        process_k,
        background_id=background_id,
    )
    (
        data_properties_dir["n_z"],
        data_properties_dir["n_y"],
        data_properties_dir["n_x"],
    ) = calculate_max_eval_image_size(
        data_dir=data_dir,
        project_name=project_name,
        test_name=test_name,
        mode=mode,
        anisotropy_factor=anisotropy_factor,
    )
    data_properties_dir["one_hot"] = one_hot
    data_properties_dir[
        "avg_background_intensity"
    ] = calculate_avg_background_intensity(
        data_dir, project_name, train_val_name, one_hot, background_id=background_id
    )
    data_properties_dir["project_name"] = project_name
    return data_properties_dir
