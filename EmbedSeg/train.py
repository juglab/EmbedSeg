import os
import shutil

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.criterions import get_loss
from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import (
    AverageMeter,
    Cluster,
    Cluster_3d,
    Logger,
    Visualizer,
    prepare_embedding_for_train_image,
)

import numpy as np

torch.backends.cudnn.benchmark = True


def train(virtual_batch_multiplier, one_hot, n_sigma, args, device):
    """Trains 2D Model with virtual multiplier

    Virtual batching code inspired from:
    https://medium.com/huggingface/training-larger-batches-
    practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

    Parameters
    ----------
    virtual_batch_multiplier : int
        Set to 1 by default. Effective batch size is product of
        virtual_batch_multiplier and batch-size
    one_hot : bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
    n_sigma: int
        Should be equal to 2 for a 2D model
    args: dictionary

    Returns
    -------
    float
        Average loss
    """

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()
    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))

    optimizer.zero_grad()  # Reset gradients tensors
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample["image"]
        im = im.to(device)
        instances = sample["instance"].squeeze(1).to(device)
        class_labels = sample["label"].squeeze(1).to(device)
        center_images = sample["center_image"].squeeze(1).to(device)
        output = model(im)  # Forward pass
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)
        loss = loss.mean()
        loss.backward()  # Backward pass
        if (i + 1) % virtual_batch_multiplier == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
        loss_meter.update(loss.item())
    return loss_meter.avg * virtual_batch_multiplier


def train_vanilla(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    pixel_x,
    pixel_y,
    n_sigma,
    args,
    device,
):
    """Trains 2D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new training image, the corresponding GT and model
        prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0,
        then each dX or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0,
        then each dY or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    n_sigma: int
        Should be set equal to 2 for a 2D model
    args: dictionary

    Returns
    -------
    float
        Average loss
    """
    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample["image"]
        im = im.to(device)
        instances = (
            sample["instance"].squeeze(1).to(device)
        )  # 1YX (not one-hot) or 1DYX (one-hot)
        class_labels = sample["label"].squeeze(1).to(device)  # 1YX
        center_images = sample["center_image"].squeeze(1).to(device)  # 1YX
        output = model(im)  # B 5 Y X
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        if display and i % display_it == 0:
            with torch.no_grad():
                visualizer.display(im[0], key="image", title="Image")
                predictions = cluster.cluster_with_gt(
                    output[0], instances[0], n_sigma=n_sigma
                )
                if one_hot:
                    instance = invert_one_hot(instances[0].cpu().detach().numpy())
                    visualizer.display(
                        instance, key="groundtruth", title="Ground Truth"
                    )  # TODO
                    instance_ids = np.arange(instances.size(1))  # instances[0] --> DYX
                else:
                    visualizer.display(
                        instances[0].cpu(), key="groundtruth", title="Ground Truth"
                    )  # TODO
                    instance_ids = instances[0].unique()
                    instance_ids = instance_ids[instance_ids != 0]

                if display_embedding:
                    (
                        center_x,
                        center_y,
                        samples_x,
                        samples_y,
                        sample_spatial_embedding_x,
                        sample_spatial_embedding_y,
                        sigma_x,
                        sigma_y,
                        color_sample_dic,
                        color_embedding_dic,
                    ) = prepare_embedding_for_train_image(
                        one_hot=one_hot,
                        grid_x=grid_x,
                        grid_y=grid_y,
                        pixel_x=pixel_x,
                        pixel_y=pixel_y,
                        predictions=predictions,
                        instance_ids=instance_ids,
                        center_images=center_images,
                        output=output,
                        instances=instances,
                        n_sigma=n_sigma,
                    )
                    if one_hot:
                        visualizer.display(
                            torch.max(instances[0], dim=0)[0],
                            key="center",
                            title="Center",
                            center_x=center_x,
                            center_y=center_y,
                            samples_x=samples_x,
                            samples_y=samples_y,
                            sample_spatial_embedding_x=sample_spatial_embedding_x,
                            sample_spatial_embedding_y=sample_spatial_embedding_y,
                            sigma_x=sigma_x,
                            sigma_y=sigma_y,
                            color_sample=color_sample_dic,
                            color_embedding=color_embedding_dic,
                        )
                    else:
                        visualizer.display(
                            instances[0] > 0,
                            key="center",
                            title="Center",
                            center_x=center_x,
                            center_y=center_y,
                            samples_x=samples_x,
                            samples_y=samples_y,
                            sample_spatial_embedding_x=sample_spatial_embedding_x,
                            sample_spatial_embedding_y=sample_spatial_embedding_y,
                            sigma_x=sigma_x,
                            sigma_y=sigma_y,
                            color_sample=color_sample_dic,
                            color_embedding=color_embedding_dic,
                        )
                visualizer.display(
                    predictions.cpu(), key="prediction", title="Prediction"
                )  # TODO

    return loss_meter.avg


def train_3d(virtual_batch_multiplier, one_hot, n_sigma, args, device):
    """Trains 3D Model with virtual multiplier

    Virtual batching code inspired from:
    https://medium.com/huggingface/training-larger-batches-practical-tips
    -on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

    Parameters
    ----------
    virtual_batch_multiplier : int
        Set to 1 by default. Effective batch size is product
        of virtual_batch_multiplier and batch-size
    one_hot : bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data, and will
        be deprecated in a future code update
    n_sigma: int
        Should be equal to 3 for a 3D model
    args: dictionary

    device: torch.device

    Returns
    -------
    float
        Average loss
    """

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()
    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))

    optimizer.zero_grad()  # Reset gradients tensors
    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample["image"]
        im = im.to(device)
        instances = sample["instance"].squeeze(1).to(device)
        class_labels = sample["label"].squeeze(1).to(device)
        center_images = sample["center_image"].squeeze(1).to(device)
        output = model(im)  # Forward pass

        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)
        loss = loss.mean()
        loss.backward()  # Backward pass
        if (i + 1) % virtual_batch_multiplier == 0:
            # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
        loss_meter.update(loss.item())
    return loss_meter.avg * virtual_batch_multiplier


def train_vanilla_3d(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    grid_z,
    pixel_x,
    pixel_y,
    pixel_z,
    n_sigma,
    zslice,
    args,
    device,
):
    """Trains 3D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new training image, the corresponding GT
        and model prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data
        and will be deprecated in a future code update
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    grid_z: int
        Number of pixels along z dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0,
        then each dX or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0,
        then each dY or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    pixel_z: float
        The grid length along z is mapped to `pixel_z`.
        For example, if grid_z = 1024 and pixel_z = 1.0,
        then each dY or the spacing between consecutive pixels
        along the z dimension is set equal to pixel_z/grid_z = 1.0/1024
    n_sigma: int
        Should be set equal to 3 for a 3D model
    zslice: int
        If `display` = True,
        then the the raw image at z = z_slice is displayed during training
    args: dictionary

    device: torch.device

    Returns
    -------
    float
        Average loss
    """
    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print("learning rate: {}".format(param_group["lr"]))

    for i, sample in enumerate(tqdm(train_dataset_it)):
        im = sample["image"].to(device)  # BCZYX
        instances = sample["instance"].squeeze(1).to(device)  # BZYX
        class_labels = sample["label"].squeeze(1).to(device)  # BZYX
        center_images = sample["center_image"].squeeze(1).to(device)  # BZYX

        output = model(im)  # B 7 Z Y X
        loss = criterion(output, instances, class_labels, center_images, **args)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        if display and i % display_it == 0:
            with torch.no_grad():
                visualizer.display(im[0, 0, zslice], key="image", title="Image")
                predictions = cluster.cluster_with_gt(
                    output[0], instances[0], n_sigma=n_sigma
                )
                if one_hot:
                    instance = invert_one_hot(instances[0].cpu().detach().numpy())
                    visualizer.display(
                        instance, key="groundtruth", title="Ground Truth"
                    )  # TODO
                    instance_ids = np.arange(instances.size(1))  # instances[0] --> DYX
                else:
                    visualizer.display(
                        instances[0, zslice].cpu(),
                        key="groundtruth",
                        title="Ground Truth",
                    )  # TODO
                    instance_ids = instances[0].unique()
                    instance_ids = instance_ids[instance_ids != 0]

                visualizer.display(
                    predictions.cpu()[zslice, ...], key="prediction", title="Prediction"
                )  # TODO

    return loss_meter.avg


def val(virtual_batch_multiplier, one_hot, n_sigma, args, device):
    """Validates a 2D Model with virtual multiplier

    Virtual batching code inspired from:
    https://medium.com/huggingface/training-larger-batches-practical-
    tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

    Parameters
    ----------
    virtual_batch_multiplier : int
        Set to 1 by default. Effective batch size is
        product of virtual_batch_multiplier and batch_size
    one_hot : bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
    n_sigma: int
        Should be equal to 2 for a 2D model
    args: dictionary

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample["image"].to(device)
            instances = sample["instance"].squeeze(1).to(device)
            class_labels = sample["label"].squeeze(1).to(device)
            center_images = sample["center_image"].squeeze(1).to(device)
            output = model(im)
            loss = criterion(
                output,
                instances,
                class_labels,
                center_images,
                **args,
                iou=True,
                iou_meter=iou_meter
            )
            loss = loss.mean()
            loss = loss / virtual_batch_multiplier
            loss_meter.update(loss.item())

    return loss_meter.avg * virtual_batch_multiplier, iou_meter.avg


def val_vanilla(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    pixel_x,
    pixel_y,
    n_sigma,
    args,
    device,
):
    """Validates a 2D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new validation image, the corresponding
        GT and model prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0,
        then each dX or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0,
        then each dY or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    n_sigma: int
        Should be set equal to 2 for a 2D model
    args: dictionary

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample["image"].to(device)
            instances = sample["instance"].squeeze(1).to(device)
            class_labels = sample["label"].squeeze(1).to(device)
            center_images = sample["center_image"].squeeze(1).to(device)
            output = model(im)
            loss = criterion(
                output,
                instances,
                class_labels,
                center_images,
                **args,
                iou=True,
                iou_meter=iou_meter
            )
            loss = loss.mean()
            if display and i % display_it == 0:
                with torch.no_grad():
                    visualizer.display(im[0], key="image", title="Image")
                    predictions = cluster.cluster_with_gt(
                        output[0], instances[0], n_sigma=n_sigma
                    )
                    if one_hot:
                        instance = invert_one_hot(instances[0].cpu().detach().numpy())
                        visualizer.display(
                            instance, key="groundtruth", title="Ground Truth"
                        )  # TODO
                        instance_ids = np.arange(instances.size(1))
                    else:
                        visualizer.display(
                            instances[0].cpu(), key="groundtruth", title="Ground Truth"
                        )  # TODO
                        instance_ids = instances[0].unique()
                        instance_ids = instance_ids[instance_ids != 0]
                    if display_embedding:
                        (
                            center_x,
                            center_y,
                            samples_x,
                            samples_y,
                            sample_spatial_embedding_x,
                            sample_spatial_embedding_y,
                            sigma_x,
                            sigma_y,
                            color_sample_dic,
                            color_embedding_dic,
                        ) = prepare_embedding_for_train_image(
                            one_hot=one_hot,
                            grid_x=grid_x,
                            grid_y=grid_y,
                            pixel_x=pixel_x,
                            pixel_y=pixel_y,
                            predictions=predictions,
                            instance_ids=instance_ids,
                            center_images=center_images,
                            output=output,
                            instances=instances,
                            n_sigma=n_sigma,
                        )
                        if one_hot:
                            visualizer.display(
                                torch.max(instances[0], dim=0)[0].cpu(),
                                key="center",
                                title="Center",
                                # torch.max returns a tuple
                                center_x=center_x,
                                center_y=center_y,
                                samples_x=samples_x,
                                samples_y=samples_y,
                                sample_spatial_embedding_x=sample_spatial_embedding_x,
                                sample_spatial_embedding_y=sample_spatial_embedding_y,
                                sigma_x=sigma_x,
                                sigma_y=sigma_y,
                                color_sample=color_sample_dic,
                                color_embedding=color_embedding_dic,
                            )
                        else:
                            visualizer.display(
                                instances[0] > 0,
                                key="center",
                                title="Center",
                                center_x=center_x,
                                center_y=center_y,
                                samples_x=samples_x,
                                samples_y=samples_y,
                                sample_spatial_embedding_x=sample_spatial_embedding_x,
                                sample_spatial_embedding_y=sample_spatial_embedding_y,
                                sigma_x=sigma_x,
                                sigma_y=sigma_y,
                                color_sample=color_sample_dic,
                                color_embedding=color_embedding_dic,
                            )

                    visualizer.display(
                        predictions.cpu(), key="prediction", title="Prediction"
                    )  # TODO

            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg


def val_3d(virtual_batch_multiplier, one_hot, n_sigma, args, device):
    """Validates a 3D Model with virtual multiplier

    Virtual batching code inspired from:
    https://medium.com/huggingface/training-larger-batches-practical-
    tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255

    Parameters
    ----------
    virtual_batch_multiplier : int
        Set to 1 by default. Effective batch size is
        product of virtual_batch_multiplier and batch-size
    one_hot : bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data,
        and will be deprecated in a future code update
    n_sigma: int
        Should be equal to 3 for a 3D model
    args: dictionary

    device: torch.device

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample["image"].to(device)
            instances = sample["instance"].squeeze(1).to(device)
            class_labels = sample["label"].squeeze(1).to(device)
            center_images = sample["center_image"].squeeze(1).to(device)
            output = model(im)
            loss = criterion(
                output,
                instances,
                class_labels,
                center_images,
                **args,
                iou=True,
                iou_meter=iou_meter
            )
            loss = loss.mean()
            loss = loss / virtual_batch_multiplier
            loss_meter.update(loss.item())

    return loss_meter.avg * virtual_batch_multiplier, iou_meter.avg


def val_vanilla_3d(
    display,
    display_embedding,
    display_it,
    one_hot,
    grid_x,
    grid_y,
    grid_z,
    pixel_x,
    pixel_y,
    pixel_z,
    n_sigma,
    zslice,
    args,
    device,
):
    """Validates a 3D Model without virtual multiplier.

    Parameters
    ----------
    display : bool
        Displays input, GT, model predictions during training
    display_embedding : bool
        Displays embeddings for train (crop) images
    display_it: int
        Displays a new training image, the corresponding GT
        and model prediction every `display_it` crop images
    one_hot: bool
        In case the GT labels are available in one-hot fashion,
        this parameter is set equal to True
        This parameter is not relevant for 3D data
        and will be deprecated in a future code update
    grid_x: int
        Number of pixels along x dimension which constitute a tile
    grid_y: int
        Number of pixels along y dimension which constitute a tile
    grid_z: int
        Number of pixels along z dimension which constitute a tile
    pixel_x: float
        The grid length along x is mapped to `pixel_x`.
        For example, if grid_x = 1024 and pixel_x = 1.0, then each dX
        or the spacing between consecutive pixels
        along the x dimension is set equal to pixel_x/grid_x = 1.0/1024
    pixel_y: float
        The grid length along y is mapped to `pixel_y`.
        For example, if grid_y = 1024 and pixel_y = 1.0, then each dY
        or the spacing between consecutive pixels
        along the y dimension is set equal to pixel_y/grid_y = 1.0/1024
    pixel_z: float
        The grid length along z is mapped to `pixel_z`.
        For example, if grid_z = 1024 and pixel_z = 1.0, then each dY
        or the spacing between consecutive pixels
        along the z dimension is set equal to pixel_z/grid_z = 1.0/1024
    n_sigma: int
        Should be set equal to 3 for a 3D model
    zslice: int
        If `display` = True, then the the raw image at z = z_slice
        is displayed during training
    args: dictionary

    device: torch.device

    Returns
    -------
    tuple: (float, float)
        Average loss, Average IoU
    """
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            im = sample["image"].to(device)  # BCZYX
            instances = sample["instance"].squeeze(1).to(device)  # BZYX
            class_labels = sample["label"].squeeze(1).to(device)  # BZYX
            center_images = sample["center_image"].squeeze(1).to(device)  # BZYX
            output = model(im)
            loss = criterion(
                output,
                instances,
                class_labels,
                center_images,
                **args,
                iou=True,
                iou_meter=iou_meter
            )
            loss = loss.mean()
            if display and i % display_it == 0:
                with torch.no_grad():
                    visualizer.display(im[0, 0, zslice], key="image", title="Image")
                    predictions = cluster.cluster_with_gt(
                        output[0], instances[0], n_sigma=n_sigma
                    )
                    if one_hot:
                        instance = invert_one_hot(instances[0].cpu().detach().numpy())
                        visualizer.display(
                            instance, key="groundtruth", title="Ground Truth"
                        )  # TODO
                        instance_ids = np.arange(instances.size(1))
                    else:
                        visualizer.display(
                            instances[0, zslice].cpu(),
                            key="groundtruth",
                            title="Ground Truth",
                        )  # TODO
                        instance_ids = instances[0].unique()
                        instance_ids = instance_ids[instance_ids != 0]

                    visualizer.display(
                        predictions.cpu()[zslice, ...],
                        key="prediction",
                        title="Prediction",
                    )  # TODO

            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg


def invert_one_hot(image):
    """Inverts a one-hot label mask.

    Parameters
    ----------
    image : numpy array (I x H x W)
        Label mask present in one-hot fashion
        (i.e. with 0s and 1s and multiple z slices)
        here `I` is the number of GT or predicted objects

    Returns
    -------
    numpy array (H x W)
        A flattened label mask with objects labelled from 1 ... I
    """
    instance = np.zeros((image.shape[1], image.shape[2]), dtype="uint16")
    for z in range(image.shape[0]):
        instance = np.where(image[z] > 0, instance + z + 1, instance)
        # TODO - Alternate ways of inverting one-hot label masks would exist !!
    return instance


def save_checkpoint(
    state, is_best, epoch, save_dir, save_checkpoint_frequency, name="checkpoint.pth"
):
    """Trains 3D Model without virtual multiplier.

    Parameters
    ----------
    state : dictionary
        The state of the model weights
    is_best : bool
        In case the validation IoU is higher at the end of a certain epoch
        than previously recorded, `is_best` is set equal to True
    epoch: int
        The current epoch
    save_checkpoint_frequency: int
        The model weights are saved every `save_checkpoint_frequency` epochs
    name: str, optional
        The model weights are saved under the name `name`

    Returns
    -------

    """
    print("=> saving checkpoint")
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if save_checkpoint_frequency is not None:
        if epoch % int(save_checkpoint_frequency) == 0:
            file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
            torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_dir, "best_iou_model.pth"))


def begin_training(
    train_dataset_dict,
    val_dataset_dict,
    model_dict,
    loss_dict,
    configs,
    color_map="magma",
):
    """Entry function for beginning the model training procedure.

    Parameters
    ----------
    train_dataset_dict : dictionary
        Dictionary containing training data loader-specific parameters
        (for e.g. train_batch_size etc)
    val_dataset_dict : dictionary
        Dictionary containing validation data loader-specific parameters
        (for e.g. val_batch_size etc)
    model_dict: dictionary
        Dictionary containing model specific parameters (for e.g. number of outputs)
    loss_dict: dictionary
        Dictionary containing loss specific parameters
        (for e.g. convex weights of different loss terms - w_iou, w_var etc)
    configs: dictionary
        Dictionary containing general training parameters
        (for e.g. num_epochs, learning_rate etc)
    color_map: str, optional
       Name of color map. Used in case configs['display'] is set equal to True

    Returns
    -------
    """

    if configs["save"]:
        if not os.path.exists(configs["save_dir"]):
            os.makedirs(configs["save_dir"])

    if configs["display"]:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")

    # set device
    device = torch.device(configs["device"])

    # define global variables
    global train_dataset_it, val_dataset_it, model, criterion, optimizer, visualizer, cluster

    # train dataloader

    train_dataset = get_dataset(
        train_dataset_dict["name"], train_dataset_dict["kwargs"]
    )
    train_dataset_it = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=train_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    # val dataloader
    val_dataset = get_dataset(val_dataset_dict["name"], val_dataset_dict["kwargs"])
    val_dataset_it = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_dataset_dict["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=val_dataset_dict["workers"],
        pin_memory=True if configs["device"][:4] == "cuda" else False,
    )

    # set model
    model = get_model(model_dict["name"], model_dict["kwargs"])
    model.init_output(loss_dict["lossOpts"]["n_sigma"])
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    if configs["grid_z"] is None:
        criterion = get_loss(
            grid_z=None,
            grid_y=configs["grid_y"],
            grid_x=configs["grid_x"],
            pixel_z=None,
            pixel_y=configs["pixel_y"],
            pixel_x=configs["pixel_x"],
            one_hot=configs["one_hot"],
            loss_opts=loss_dict["lossOpts"],
        )
    else:
        criterion = get_loss(
            configs["grid_z"],
            configs["grid_y"],
            configs["grid_x"],
            configs["pixel_z"],
            configs["pixel_y"],
            configs["pixel_x"],
            configs["one_hot"],
            loss_dict["lossOpts"],
        )
    criterion = criterion.to(device)
    # criterion = torch.nn.DataParallel(criterion).to(device)

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configs["train_lr"], weight_decay=1e-4
    )

    def lambda_(epoch):
        return pow((1 - ((epoch) / 200)), 0.9)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )

    if configs["grid_z"] is None:
        # clustering
        cluster = Cluster(
            configs["grid_y"],
            configs["grid_x"],
            configs["pixel_y"],
            configs["pixel_x"],
            device,
            configs["one_hot"],
        )
    else:
        # clustering
        cluster = Cluster_3d(
            configs["grid_z"],
            configs["grid_y"],
            configs["grid_x"],
            configs["pixel_z"],
            configs["pixel_y"],
            configs["pixel_x"],
            device,
            configs["one_hot"],
        )

    # Visualizer
    visualizer = Visualizer(
        ("image", "groundtruth", "prediction", "center"), color_map
    )  # 5 keys

    # Logger
    logger = Logger(("train", "val", "iou"), "loss")

    # resume
    start_epoch = 0
    best_iou = 0
    if configs["resume_path"] is not None and os.path.exists(configs["resume_path"]):
        print("Resuming model from {}".format(configs["resume_path"]))
        state = torch.load(configs["resume_path"])
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    for epoch in range(start_epoch, configs["n_epochs"]):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=epoch - 1
        )
        print("Starting epoch {}".format(epoch))
        # scheduler.step(epoch)

        if configs["grid_z"] is None:
            if train_dataset_dict["virtual_batch_multiplier"] > 1:
                train_loss = train(
                    virtual_batch_multiplier=train_dataset_dict[
                        "virtual_batch_multiplier"
                    ],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    args=loss_dict["lossW"],
                    device=device,
                )
            elif train_dataset_dict["virtual_batch_multiplier"] == 1:
                train_loss = train_vanilla(
                    display=configs["display"],
                    display_embedding=configs["display_embedding"],
                    display_it=configs["display_it"],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    grid_x=configs["grid_x"],
                    grid_y=configs["grid_y"],
                    pixel_x=configs["pixel_x"],
                    pixel_y=configs["pixel_y"],
                    args=loss_dict["lossW"],
                    device=device,
                )

            if val_dataset_dict["virtual_batch_multiplier"] > 1:
                val_loss, val_iou = val(
                    virtual_batch_multiplier=val_dataset_dict[
                        "virtual_batch_multiplier"
                    ],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    args=loss_dict["lossW"],
                    device=device,
                )
            elif val_dataset_dict["virtual_batch_multiplier"] == 1:
                val_loss, val_iou = val_vanilla(
                    display=configs["display"],
                    display_embedding=configs["display_embedding"],
                    display_it=configs["display_it"],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    grid_x=configs["grid_x"],
                    grid_y=configs["grid_y"],
                    pixel_x=configs["pixel_x"],
                    pixel_y=configs["pixel_y"],
                    args=loss_dict["lossW"],
                    device=device,
                )
        else:
            if train_dataset_dict["virtual_batch_multiplier"] > 1:
                train_loss = train_3d(
                    virtual_batch_multiplier=train_dataset_dict[
                        "virtual_batch_multiplier"
                    ],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    args=loss_dict["lossW"],
                    device=device,
                )
            elif train_dataset_dict["virtual_batch_multiplier"] == 1:
                train_loss = train_vanilla_3d(
                    display=configs["display"],
                    display_embedding=configs["display_embedding"],
                    display_it=configs["display_it"],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    zslice=configs["display_zslice"],
                    grid_x=configs["grid_x"],
                    grid_y=configs["grid_y"],
                    grid_z=configs["grid_z"],
                    pixel_x=configs["pixel_x"],
                    pixel_y=configs["pixel_y"],
                    pixel_z=configs["pixel_z"],
                    args=loss_dict["lossW"],
                    device=device,
                )

            if val_dataset_dict["virtual_batch_multiplier"] > 1:
                val_loss, val_iou = val_3d(
                    virtual_batch_multiplier=val_dataset_dict[
                        "virtual_batch_multiplier"
                    ],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    args=loss_dict["lossW"],
                    device=device,
                )
            elif val_dataset_dict["virtual_batch_multiplier"] == 1:
                val_loss, val_iou = val_vanilla_3d(
                    display=configs["display"],
                    display_embedding=configs["display_embedding"],
                    display_it=configs["display_it"],
                    one_hot=configs["one_hot"],
                    n_sigma=loss_dict["lossOpts"]["n_sigma"],
                    zslice=configs["display_zslice"],
                    grid_x=configs["grid_x"],
                    grid_y=configs["grid_y"],
                    grid_z=configs["grid_z"],
                    pixel_x=configs["pixel_x"],
                    pixel_y=configs["pixel_y"],
                    pixel_z=configs["pixel_z"],
                    args=loss_dict["lossW"],
                    device=device,
                )

        scheduler.step()
        print("===> train loss: {:.2f}".format(train_loss))
        print("===> val loss: {:.2f}, val iou: {:.2f}".format(val_loss, val_iou))

        logger.add("train", train_loss)
        logger.add("val", val_loss)
        logger.add("iou", val_iou)
        logger.plot(save=configs["save"], save_dir=configs["save_dir"])  # TODO

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if configs["save"]:
            state = {
                "epoch": epoch,
                "best_iou": best_iou,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
        save_checkpoint(
            state,
            is_best,
            epoch,
            save_dir=configs["save_dir"],
            save_checkpoint_frequency=configs["save_checkpoint_frequency"],
        )
