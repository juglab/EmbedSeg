import os
import torch

import EmbedSeg.utils.transforms as my_transforms


def create_dataset_dict(
    data_dir,
    project_name,
    size,
    center,
    type,
    one_hot=False,
    name="2d",
    batch_size=16,
    virtual_batch_multiplier=1,
    workers=8,
):
    """
    Creates `dataset_dict` dictionary from parameters.
    Parameters
    ----------
    data_dir: string
        Data is read from os.path.join(data_dir, project_name)
    project_name: string
        Data is read from os.path.join(data_dir, project_name)
    size: int
        Number of image-mask per epoch
    center: string
        One of 'centroid', 'approximate-medoid', 'medoid'
    type: string
        One of 'train', 'val'
    one_hot: boolean
        If 'True', instance images are perceived as
        DYX (here each object is encoded as 1 in its individual slice)
        If 'False', instance image is perceived as
        YX and has the same dimensionality as raw image
    name: string
        One of '2d' or '3d'
    batch_size: int
        Effective Batch-size is the product of `batch_size`
        and `virtual_batch_multiplier`
    virtual_batch_multiplier: int
        Effective Batch-size is the product of `batch_size`
        and `virtual_batch_multiplier`
    workers: int
        Number of data-loader workers
    """
    if name == "2d":
        set_transforms = my_transforms.get_transform(
            [
                {
                    "name": "RandomRotationsAndFlips",
                    "opts": {
                        "keys": ("image", "instance", "label", "center_image"),
                        "degrees": 90,
                        "one_hot": one_hot,
                    },
                },
                {
                    "name": "ToTensorFromNumpy",
                    "opts": {
                        "keys": ("image", "instance", "label", "center_image"),
                        "type": (
                            torch.FloatTensor,
                            torch.ShortTensor,
                            torch.ShortTensor,
                            torch.BoolTensor,
                        ),
                    },
                },
            ]
        )
    elif name == "3d":
        set_transforms = my_transforms.get_transform(
            [
                {
                    "name": "RandomRotationsAndFlips_3d",
                    "opts": {
                        "keys": ("image", "instance", "label", "center_image"),
                        "degrees": 90,
                        "one_hot": one_hot,
                    },
                },
                {
                    "name": "ToTensorFromNumpy",
                    "opts": {
                        "keys": ("image", "instance", "label", "center_image"),
                        "type": (
                            torch.FloatTensor,
                            torch.ShortTensor,
                            torch.ShortTensor,
                            torch.BoolTensor,
                        ),
                    },
                },
            ]
        )
    dataset_dict = {
        "name": name,
        "kwargs": {
            "center": "center-" + center,
            "data_dir": os.path.join(data_dir, project_name),
            "type": type,
            "size": size,
            "transform": set_transforms,
            "one_hot": one_hot,
        },
        "batch_size": batch_size,
        "virtual_batch_multiplier": virtual_batch_multiplier,
        "workers": workers,
    }
    print(
        "`{}_dataset_dict` dictionary successfully created \
                with: \n -- {} images accessed from {}, "
        "\n -- number of images per epoch equal to {}, "
        "\n -- batch size set at {}, ".format(
            type,
            type,
            os.path.join(data_dir, project_name, type, "images"),
            size,
            batch_size,
        )
    )
    return dataset_dict


def create_test_configs_dict(
    data_dir,
    checkpoint_path,
    norm,
    data_type,
    save_dir=None,
    normalization_factor=1.0,
    tta=True,
    one_hot=False,
    ap_val=0.5,
    seed_thresh=0.9,
    fg_thresh=0.5,
    min_object_size=10,
    mean_object_size=None,
    save_images=True,
    save_results=True,
    min_mask_sum=0,
    min_unclustered_sum=0,
    device="cuda:0",
    n_z=None,
    n_y=1024,
    n_x=1024,
    num_workers=4,
    anisotropy_factor=None,
    l_y=1,
    l_x=1,
    name="2d",
    input_channels=1,
    type="test",
    normalization=True,
    cluster_fast=True,
    expand_grid=True,
    uniform_ds_factor=1,
    bg_id=0,
):
    """
    Creates `test_configs` dictionary from parameters.
    Parameters
    ----------
    data_dir : str
        Data is read from os.path.join(data_dir, 'test')
    checkpoint_path: str
        This indicates the path to the trained model
    data_type: str
        This reflects the data-type of the image and should be equal
        to one of '8-bit' or '16-bit'
    save_dir: str
        This indicates the directory where the results are saved
    normalization_factor: float, default = 1.0
        This, in general, does not need to be specified.
        It is a legacy parameter, which will be deprecated in a later release.
    tta: boolean
        If True, then use test-time-augmentation
        This parameter generally gives better results during inference,
        but increases the inference time.
    one_hot: boolean
        If True, then GT evaluation instance instance images are available
        in one-hot encoded style (DYX)
        This parameter is not applicable for a 3D setting,
        since a pixel can be allocated to only one GT 3D object instance
    ap_val: float, default = 0.5
        While computing the AP_dsb, a true positive, false positive and
        false negative is determined
        based on whether the IoU between the predicted and GT object >=ap_val
    seed_thresh: float, default = 0.9
        Only considers a subset of pixels for whom the seediness score
        is above the `seed_thresh`
        These are then considered as potential object seeds
    fg_thresh: float, default = 0.5
        Only considers a subset of pixels for whom
        the seediness score is above the `fg_thresh`
        These are then clustered into unique objects
    min_object_size: int
        Ignores objects having pixels less than min_object_size
    mean_object_size: float
        Average object size in terms of number of pixels
        This is used in case running an Integer Linear Programming
        problem for combining objects
    save_images: boolean
        If True, then prediction images are saved
    save_results: boolean
        If True, then prediction results are saved in text file
    min_mask_sum: int, default = 0
        Only start creating instances, if there are at least
        `min_mask_sum` pixels in foreground!
    min_unclustered_sum: int, default = 0
        Stop when the number of seed candidates are less than `min_unclustered_sum`
    device: string
        Default: 'cuda:0'
        Could set to 'mps' for acceleration with mac
        Other options are 'cpu'
    n_z: int
        Size of the underlying grid along the z (or depth) dimension
        Should be set to None, for a 2D setting
    n_y: int
        Size of the underlying grid along the y (or height) dimension
    n_x: int
        Size of the underlying grid along the x (or width) dimension
    anisotropy_factor: float
        Ratio of the sizes of the z and x pixel sizes
        If the Raw Image is acquired in a down-sampled manner
        along the z dimension, then `anisotropy_factor` > 1
    l_y: float
        If l_y = 1.0 and n_y = 1024, then the pixel spacing
        along the y dimension is 1.0/1023
    l_x: float
        If l_x = 1.0 and n_x = 1024, then the pixel spacing
        along the x dimension is 1.0/1023
    name: str
        One of '2d', '3d', '3d_sliced'
    input_channels: int
        Number of channels in the Raw Image
        For example, if Gray-scale, then 'input_channels' is equal to 1
        If RGB, then 'input_channels' is equal to 3
    type: str
        One of 'train', 'val' or 'test'
    normalization: bool
        One of True or False
        For test images, 'normalization' should be True
        For other crops used during training, 'normalization' should be False
        since these image crops are already normalized
    cluster_fast : bool, Default = True
        If True, then both 'fg_thresh' and 'seed_thresh' are used
        If False, then only 'fg_thresh' is used and seeds are identified as
        local maxima in the seediness map
    expand_grid: bool, default = True
        If True, then for a different sized test image,
        the grid and pixel spacing is automatically adjusted
        If False, then prediction occurs on original grid or tile,
        and these predictions are then stitched
    uniform_ds_factor: int, default = 1
        If the original crops were generated by down-sampling in '01-data.ipynb',
        then the test images
        should also be down-sampled before being input to the model
        and the predictions should then be up-sampled
        to restore them back to the original size
    bg_id : int, default = 0
        Label of background in the ground truth test label masks
        This parameter is only used while quantifying accuracy predicted label
        masks with ground truth label masks
    num_workers: int, default = 4
    """
    if n_z is None:
        l_z = None
    else:
        l_z = (n_z - 1) / (n_x - 1) * anisotropy_factor

    if name == "2d":
        n_sigma = 2
        num_classes = [4, 1]
        model_name = "branched_erfnet"
    elif name == "3d":
        n_sigma = 3
        num_classes = [6, 1]
        model_name = "branched_erfnet_3d"
    elif name == "3d_sliced":
        n_sigma = 3
        num_classes = [4, 1]
        model_name = "branched_erfnet"
    elif name == "3d_ilp":
        n_sigma = 2
        num_classes = [4, 1]
        model_name = "branched_erfnet"

    if name in ["3d_sliced"]:
        sliced_mode = True
    else:
        sliced_mode = False

    test_configs = dict(
        ap_val=ap_val,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
        mean_object_size=mean_object_size,
        n_sigma=n_sigma,
        tta=tta,
        seed_thresh=seed_thresh,
        fg_thresh=fg_thresh,
        device=device,
        save_results=save_results,
        save_images=save_images,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
        grid_x=n_x,
        grid_y=n_y,
        grid_z=n_z,
        pixel_x=l_x,
        pixel_y=l_y,
        pixel_z=l_z,
        name=name,
        num_workers=num_workers,
        anisotropy_factor=anisotropy_factor,
        cluster_fast=cluster_fast,
        expand_grid=expand_grid,
        uniform_ds_factor=uniform_ds_factor,
        dataset={
            "name": name,
            "kwargs": {
                "data_dir": data_dir,
                "type": type,
                "data_type": data_type,
                "norm": norm,
                "sliced_mode": sliced_mode,
                "anisotropy_factor": anisotropy_factor,
                "normalization": normalization,
                "uniform_ds_factor": uniform_ds_factor,
                "bg_id": bg_id,
                "transform": my_transforms.get_transform(
                    [
                        {
                            "name": "ToTensorFromNumpy",
                            "opts": {
                                "keys": ("image", "instance", "label"),
                                "type": (
                                    torch.FloatTensor,
                                    torch.ShortTensor,
                                    torch.ShortTensor,
                                ),
                                "normalization_factor": normalization_factor,
                            },
                        },
                    ]
                ),
                "one_hot": one_hot,
            },
        },
        model={
            "name": model_name,
            "kwargs": {
                "input_channels": input_channels,
                "num_classes": num_classes,
            },
        },
    )
    print(
        "`test_configs` dictionary successfully created with: "
        "\n -- evaluation images accessed from {}, "
        "\n -- trained weights accessed from {}, "
        "\n -- output directory chosen as {}".format(
            data_dir, checkpoint_path, save_dir
        )
    )
    return test_configs


def create_model_dict(input_channels, num_classes=[4, 1], name="2d"):
    """
    Creates `model_dict` dictionary from parameters.
    Parameters
    ----------
    input_channels: int
        1 indicates gray-channle image, 3 indicates RGB image.
    num_classes: list
        [4, 1] -> 4 indicates offset in x, offset in y,
        margin in x, margin in y; 1 indicates seediness score
    name: string
    """
    model_dict = {
        "name": "branched_erfnet" if name == "2d" else "branched_erfnet_3d",
        "kwargs": {
            "num_classes": num_classes,
            "input_channels": input_channels,
        },
    }
    print(
        "`model_dict` dictionary successfully created \
                with: \n -- num of classes equal to {}, \n -- input channels \
                equal to {}, \n -- name equal to {}".format(
            input_channels, num_classes, model_dict["name"]
        )
    )
    return model_dict


def create_loss_dict(foreground_weight=10, n_sigma=2, w_inst=1, w_var=10, w_seed=1):
    """
    Creates `loss_dict` dictionary from parameters.
    Parameters
    ----------
    foreground_weight: int
    w_inst: int/float
        weight on IOU loss
    w_var: int/float
        weight on variance loss
    w_seed: int/float
        weight on seediness loss
    """
    loss_dict = {
        "lossOpts": {
            "n_sigma": n_sigma,
            "foreground_weight": foreground_weight,
        },
        "lossW": {
            "w_inst": w_inst,
            "w_var": w_var,
            "w_seed": w_seed,
        },
    }
    print(
        "`loss_dict` dictionary successfully created \
                with: \n -- foreground weight equal to {:.3f}, \n -- w_inst \
                equal to {}, \n -- w_var \
                equal to {}, \n -- w_seed equal to {}".format(
            foreground_weight, w_inst, w_var, w_seed
        )
    )
    return loss_dict


def create_configs(
    save_dir,
    resume_path,
    one_hot=False,
    display=False,
    display_embedding=False,
    display_it=5,
    n_epochs=200,
    train_lr=5e-4,
    device="cuda:0",
    save=True,
    n_z=None,
    n_y=1024,
    n_x=1024,
    anisotropy_factor=None,
    l_y=1,
    l_x=1,
    save_checkpoint_frequency=None,
    display_zslice=None,
):
    """
    Creates `configs` dictionary from parameters.
    Parameters
    ----------
    save_dir: str
        Path to where the experiment is saved
    resume_path: str
        Path to where the trained model (for example, checkpoint.pth) lives
    one_hot: boolean
        If 'True', instance images are perceived as DYX
        (here each object is encoded as 1 in its individual slice)
        If 'False', instance image is perceived as YX and
        has the same dimensionality as raw image
    display: boolean
        If 'True', then realtime display of images, ground-truth,
        predictions are shown
    display_embedding: boolean
        If False, it suppresses embedding image
    display_it: int
        Shows display every n training/val steps (display_it = n)
    n_epochs: int
        Total number of epochs
    train_lr: float
        Starting learning rate
    device: string
        Default - 'cuda:0'
        Set to 'mps' to train on Accellerated PyTorch training on Mac
    save: boolean
        If True, then results are saved
    grid_y: int
        Size in y dimension of the largest evaluation image
    grid_x: int
        Size in x dimension of the largest evaluation image
    pixel_y: int
        Pixel size in y
    pixel_x: int
        Pixel size in x
    save_checkpoint_frequency: int
        Save model weights after 'n' epochs (in addition to last and best model weights)
        Default is None
    """
    if n_z is None:
        l_z = None
    else:
        l_z = (n_z - 1) / (n_x - 1) * anisotropy_factor

    configs = dict(
        train_lr=train_lr,
        n_epochs=n_epochs,
        device=device,
        display=display,
        display_embedding=display_embedding,
        display_it=display_it,
        save=save,
        save_dir=save_dir,
        resume_path=resume_path,
        grid_z=n_z,
        grid_y=n_y,
        grid_x=n_x,
        pixel_z=l_z,
        pixel_y=l_y,
        pixel_x=l_x,
        one_hot=one_hot,
        save_checkpoint_frequency=save_checkpoint_frequency,
        display_zslice=display_zslice,
    )
    print(
        "`configs` dictionary successfully created with: "
        "\n -- n_epochs equal to {}, "
        "\n -- save_dir equal to {}, "
        "\n -- n_z equal to {}, "
        "\n -- n_y equal to {}, "
        "\n -- n_x equal to {}, ".format(n_epochs, save_dir, n_z, n_y, n_x)
    )
    return configs
