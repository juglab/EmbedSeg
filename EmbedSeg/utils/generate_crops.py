import os

import numpy as np
import pandas as pd
import pycocotools.mask as rletools
import tifffile
from numba import jit
from scipy.ndimage import zoom
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm


def _fill_label_holes(lbl_img, **kwargs):
    """
    Function taken from Stardist repository: https://github.com/stardist/stardist
    """
    lbl_img_filled = np.zeros_like(lbl_img)
    for lbl in set(np.unique(lbl_img)) - set([0]):
        mask = lbl_img == lbl
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = lbl
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """
    Fill small holes in label image.
    Function taken from Stardist repository: https://github.com/stardist/stardist
    """

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def normalize_min_max_percentile(
    x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=float
):
    """
    Percentile-based image normalization.
    Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=float):
    """
    Percentile-based image normalization.
    Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalize_mean_std(x, axis=None):
    """Normalize image based on mean and standard deviation of intensity

    Parameters
    ----------
    x : numpy array
        Input Image
    axis: int
        Set equal to 0, 1, 2 ... in case you normalize based on dimension
        Can be useful if the image contains multiple channels and each channel
        should be independently normalized
    Returns
    -------
    numpy array: float
        Normalized image with mean equal to 0 and standard deviation equal to 1
    """
    x = x.astype(float)
    if axis is None:
        mean = np.mean(x)
        std = np.std(x)
    else:
        mean = np.mean(x, axis=axis)  # (C,)
        std = np.std(x, axis=axis)  # (C,)
        mean = np.expand_dims(mean, axis=axis)  # (C, H, W)
        std = np.expand_dims(std, axis=axis)  # (C, H, W)
    return (x - mean) / std


@jit(nopython=True)
def pairwise_python(X):
    """Helper function to compute pairwise Euclidean distance for a matrix
    of size M x N containing M,  N-dimensional row entries

    Parameters
    ----------
    x : numpy array
        Input Image
    Returns
    -------
    D : numpy array, M x M
        D[i, j] corresponds to the Euclidean Distance between
        the i th and j th rows of X
    """
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D


def generate_center_image(instance, center, ids, one_hot):
    """
    Generates a `center_image` which is one (True) for all center locations
    and zero (False) otherwise.

    Parameters
    ----------
    instance: numpy array
        `instance` image containing unique `ids` for each object (YX)
         or present in a one-hot encoded style where each object is one
         in its own slice and zero elsewhere.
    center: string
        One of 'centroid', 'approximate-medoid' or 'medoid'.
    ids: list
        Unique ids corresponding to the objects present in the instance image.
    one_hot: boolean
        True (in this case, `instance` has shape DYX) or False
        (in this case, `instance` has shape YX).

    Returns
    -------
    numpy array: bool
    Center image with center locations set to True
    """

    if not one_hot:
        center_image = np.zeros(instance.shape, dtype=bool)
    else:
        center_image = np.zeros((instance.shape[-2], instance.shape[-1]), dtype=bool)
    for j, id in enumerate(ids):
        if not one_hot:
            y, x = np.where(instance == id)
        else:
            y, x = np.where(instance[id] == 1)
        if len(y) != 0 and len(x) != 0:
            if center == "centroid":
                ym, xm = np.mean(y), np.mean(x)
            elif center == "approximate-medoid":
                ym_temp, xm_temp = np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
                ym, xm = y[imin], x[imin]
            elif center == "medoid":
                dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
            center_image[int(np.round(ym)), int(np.round(xm))] = True
    return center_image


def generate_center_image_3d(
    instance, center, ids, one_hot, anisotropy_factor, speed_up
):
    """
    Generates a `center_image` for 3D image crops which is one (True)
    for all center locations and zero (False) otherwise.

    Parameters
    ----------
    instance: numpy array
        `instance` image containing unique `ids` for each object (ZYX)
    center: string
        One of 'centroid', 'approximate-medoid' or 'medoid'.
    ids: list
        Unique ids corresponding to the objects present in the instance image.
    one_hot: boolean
        This parameter is not used in the 3D setting and will be deprecated.
    speed_up: int
        This computes the centers of crops faster by down-sampling crops
        along x and y dimensions.

    Returns
    -------
    numpy array: bool
    Center image with center locations set to True
    """

    center_image = np.zeros(instance.shape, dtype=bool)
    instance_downsampled = instance[
        :, :: int(speed_up), :: int(speed_up)
    ]  # down sample in x and y
    for j, id in enumerate(ids):
        z, y, x = np.where(instance_downsampled == id)
        if len(y) != 0 and len(x) != 0:
            if center == "centroid":
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
            elif center == "approximate-medoid":
                zm_temp, ym_temp, xm_temp = np.median(z), np.median(y), np.median(x)
                imin = np.argmin(
                    (x - xm_temp) ** 2
                    + (y - ym_temp) ** 2
                    + (anisotropy_factor * (z - zm_temp)) ** 2
                )
                zm, ym, xm = z[imin], y[imin], x[imin]
            elif center == "medoid":
                dist_matrix = pairwise_python(
                    np.vstack(
                        (speed_up * x, speed_up * y, anisotropy_factor * z)
                    ).transpose()
                )
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                zm, ym, xm = z[imin], y[imin], x[imin]
            center_image[
                int(np.round(zm)),
                int(np.round(speed_up * ym)),
                int(np.round(speed_up * xm)),
            ] = True
    return center_image


def sparsify(instance):
    """
    Ensures that no slice in the `instance` image crop is empty (completely zero)

    Parameters
    ----------
    instance: numpy array,
        `instance` image present in a one-hot encoded style where each object
        is one in it own slice and zero elsewhere.

    Returns
    ----------
    numpy array
    Numpy array with number of slices which are fewer or equal to
    the number of slices in `instance`
    """
    instance_sparse = []
    for z in range(instance.shape[0]):
        if np.sum(instance[z, ...]) > 0:
            instance_sparse.append(instance[z, ...])
    return np.array(instance_sparse)


def encode(filename, img, one_hot=False):
    """Convert label mask (img) in a run-length encoded fashion and save to a csv file

    Parameters
    ----------
    filename : str
        Name of csv file
    img: numpy array
        Label mask image containing instance segmentations
    one_hot: boolean, optional
        If the Label mask is present in a one-hot encoded fashion,
        set this equal to True
    Returns
    -------
    """
    data = []

    if one_hot:
        ids = np.arange(img.shape[0])
        for id in ids:
            d = rletools.encode(np.asfortranarray(img[id]) == 1)
            data.append([id, d["counts"], d["size"]])
    else:
        ids = np.unique(img)
        ids = ids[ids != 0]
        for id in ids:
            d = rletools.encode(np.asfortranarray(img) == id)
            data.append([id, d["counts"], d["size"]])
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=None)


def process(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size,
    center,
    norm="min-max-percentile",
    one_hot=False,
    data_type="8-bit",
    normalization_factor=None,
    rle_encode=False,
    fraction_max_ids=1.0,
    background_id=0,
):
    """Entry function which generates 2D crops from 2D images (one_hot is False)

    Parameters
    ----------
    im: numpy array
        raw image
    inst: numpy array
        label mask
    crops_dir: str
        Path to where the crops are saved
    data_subset: str
        Set equal to one of `train` or `val`
    crop_size: int
        Height and Width of crop
    center: str
        Set equal to one of `centroid`, `medoid` or `approximate-medoid`
    norm: str
        Set equal to one of `min-max-percentile` or `absolute` or `mean-std`
    one_hot: bool
        If label masks are available in one-hot
        encoded fashion, set `one_hot` equal to True
    data_type: str
        Set equal to `8-bit` or `16-bit`
    normalization_factor: int
        In case, norm is set equal to `absolute`, then the image intensities
        are divided by the `normalization_factor`
    rle_encode: bool
        If set equal to True, the label masks are saved as csv files
    fraction_max_ids: float, between 0 and 1.0
        If set equal to a value less than 1.0, then only that fraction of ids
        are processed are used to make crops
    background_id: int, optional
        Id of the background in the label mask

    Returns
    -------
    """

    image_path = os.path.join(crops_dir, data_subset, "images/")
    instance_path = os.path.join(crops_dir, data_subset, "masks/")
    center_image_path = os.path.join(crops_dir, data_subset, "center-" + center + "/")

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))
    if not os.path.exists(center_image_path):
        os.makedirs(os.path.dirname(center_image_path))
        print("Created new directory : {}".format(center_image_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(float)

    if norm == "min-max-percentile":
        if image.ndim == 2:  # gray-scale
            image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))
        elif image.ndim == 3:  # multi-channel image (C, H, W)
            image = normalize_min_max_percentile(image, 1, 99.8, axis=(1, 2))
    elif norm == "mean-std":
        if image.ndim == 2:
            image = normalize_mean_std(image)  # axis == None
        elif image.ndim == 3:
            image = normalize_mean_std(image, axis=(1, 2))
    elif norm == "absolute":
        if data_type == "8-bit":
            if normalization_factor is None:
                image /= 255
            else:
                image /= normalization_factor
        elif data_type == "16-bit":
            if normalization_factor is None:
                image /= 65535
            else:
                image /= normalization_factor

    instance = fill_label_holes(instance)

    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        c, h, w = image.shape
    instance_np = np.array(instance, copy=False)

    object_mask = instance_np > background_id
    # ensure that background is mapped to 0
    instance_np[instance_np == background_id] = 0
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]
    ids_subset = np.random.choice(ids, int(fraction_max_ids * len(ids)), replace=False)

    # loop over instances
    for j, id in enumerate(ids_subset):
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)

        jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
        ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))
        if image.ndim == 2:
            if image[jj : jj + crop_size, ii : ii + crop_size].shape == (
                crop_size,
                crop_size,
            ):
                im_crop = image[jj : jj + crop_size, ii : ii + crop_size]
                instance_crop = instance_np[jj : jj + crop_size, ii : ii + crop_size]
                center_image_crop = generate_center_image(
                    instance_crop, center, ids, one_hot
                )
                tifffile.imsave(
                    image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                    im_crop,
                )
                if rle_encode:
                    encode(
                        instance_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.csv".format(j),
                        instance_crop,
                    )
                    encode(
                        center_image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.csv".format(j),
                        center_image_crop,
                    )
                else:
                    tifffile.imsave(
                        instance_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.tif".format(j),
                        instance_crop,
                    )
                    tifffile.imsave(
                        center_image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.tif".format(j),
                        center_image_crop,
                    )
        elif image.ndim == 3:
            if image[:, jj : jj + crop_size, ii : ii + crop_size].shape == (
                c,
                crop_size,
                crop_size,
            ):
                im_crop = image[:, jj : jj + crop_size, ii : ii + crop_size]
                instance_crop = instance[jj : jj + crop_size, ii : ii + crop_size]
                center_image_crop = generate_center_image(
                    instance_crop, center, ids, one_hot
                )
                tifffile.imsave(
                    image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                    im_crop,
                )
                if rle_encode:
                    encode(
                        instance_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.csv".format(j),
                        instance_crop,
                    )
                    encode(
                        center_image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.csv".format(j),
                        center_image_crop,
                    )
                else:
                    tifffile.imsave(
                        instance_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.tif".format(j),
                        instance_crop,
                    )
                    tifffile.imsave(
                        center_image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}.tif".format(j),
                        center_image_crop,
                    )


def process_3d(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size_x,
    crop_size_y,
    crop_size_z,
    center,
    norm="min-max-percentile",
    one_hot=False,
    anisotropy_factor=1.0,
    speed_up=1.0,
    data_type="8-bit",
    normalization_factor=None,
    rle_encode=False,
    fraction_max_ids=1.0,
    background_id=0,
    uniform_ds_factor=1,
):
    """Entry function which generates 3D crops from 3D images

    Parameters
    ----------
    im: numpy array
        raw image
    inst: numpy array
        label mask
    crops_dir: str
        Path to where the crops are saved
    data_subset: str
        Set equal to one of `train` or `val`
    crop_size_x: int
        Width of crop
    crop_size_y: int
        Height of crop
    crop_size_z: int
        Depth of crop
    center: str
        Set equal to one of `centroid`, `medoid` or `approximate-medoid`
    norm: str
        Set equal to one of `min-max-percentile` or `absolute` or `mean-std`
    one_hot: bool
        If label masks are available in one-hot encoded fashion,
        set `one_hot` equal to True
        This parameter should be always set equal to False for 3D processing
        and will be deprecated in a later release
    anisotropy_factor: float
        This parameter should be set equal to the ratio of the z pixel size
        to the x or y pixel size
        Here, we assume that the x or y pixel size is the same
        (If the imaging is down-sampled in the z dimension,
         `anisotropy_factor` is greater than 1.0)
    data_type: str
        Set equal to `8-bit` or `16-bit`
    normalization_factor: int
        In case, norm is set equal to `absolute`,
        then the image intensities are divided by the `normalization_factor`
    rle_encode: bool
        If set equal to True, the label masks are saved as csv files
    fraction_max_ids: float, between 0 and 1.0
        If set equal to a value less than 1.0, then only that fraction of ids
        are processed are used to make crops
    background_id: int, optional
        Id of the background in the label
    uniform_ds_factor: int, optional
        In case, the image and corresponding GT instance should be down-sampled
        This serves the purpose of increasing the receptive field
        without increasing the GPU memory requirement

    Returns
    -------
    """

    image_path = os.path.join(crops_dir, data_subset, "images/")
    instance_path = os.path.join(crops_dir, data_subset, "masks/")
    center_image_path = os.path.join(crops_dir, data_subset, "center-" + center + "/")

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))
    if not os.path.exists(center_image_path):
        os.makedirs(os.path.dirname(center_image_path))
        print("Created new directory : {}".format(center_image_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(float)

    if norm == "min-max-percentile":
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
    elif norm == "mean-std":
        image = normalize_mean_std(image)
    elif norm == "absolute":
        if data_type == "8-bit":
            if normalization_factor is None:
                image /= 255
            else:
                image /= normalization_factor
        elif data_type == "16-bit":
            if normalization_factor is None:
                image /= 65535
            else:
                image /= normalization_factor
    instance = fill_label_holes(instance)

    # sometimes it helps to downsample the image and instance,
    # in order to increase the receptive field
    instance_ds = instance[
        ::uniform_ds_factor, ::uniform_ds_factor, ::uniform_ds_factor
    ]
    image_ds = image[::uniform_ds_factor, ::uniform_ds_factor, ::uniform_ds_factor]

    # but we would still like the downsampled image to be of the same
    # size as the original image, which we ensure through padding
    dz = image.shape[0] - image_ds.shape[0]
    dy = image.shape[1] - image_ds.shape[1]
    dx = image.shape[2] - image_ds.shape[2]
    pad_width = (
        (dz // 2, dz - dz // 2),
        (dy // 2, dy - dy // 2),
        (dx // 2, dx - dx // 2),
    )

    image = np.pad(image_ds, pad_width)
    instance = np.pad(instance_ds, pad_width, "constant", constant_values=background_id)

    d, h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = instance_np > background_id
    # ensure that background is mapped to 0
    instance_np[instance_np == background_id] = 0
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]
    ids_subset = np.random.choice(ids, int(fraction_max_ids * len(ids)), replace=False)
    # loop over instances
    for j, id in enumerate(ids_subset):
        z, y, x = np.where(instance_np == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
        jj = int(np.clip(ym - crop_size_y / 2, 0, h - crop_size_y))
        ii = int(np.clip(xm - crop_size_x / 2, 0, w - crop_size_x))

        if image[
            kk : kk + crop_size_z, jj : jj + crop_size_y, ii : ii + crop_size_x
        ].shape == (crop_size_z, crop_size_y, crop_size_x):
            im_crop = image[
                kk : kk + crop_size_z, jj : jj + crop_size_y, ii : ii + crop_size_x
            ]
            instance_crop = instance_np[
                kk : kk + crop_size_z, jj : jj + crop_size_y, ii : ii + crop_size_x
            ]
            center_image_crop = generate_center_image_3d(
                instance_crop,
                center,
                ids,
                one_hot=one_hot,
                anisotropy_factor=anisotropy_factor,
                speed_up=speed_up,
            )
            tifffile.imsave(
                image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                im_crop,
            )
            if rle_encode:
                encode(
                    instance_path + os.path.basename(im)[:-4] + "_{:03d}.csv".format(j),
                    instance_crop.astype(np.uint16),
                )
                encode(
                    center_image_path
                    + os.path.basename(im)[:-4]
                    + "_{:03d}.csv".format(j),
                    center_image_crop,
                )
            else:
                tifffile.imsave(
                    instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                    instance_crop.astype(np.uint16),
                )
                tifffile.imsave(
                    center_image_path
                    + os.path.basename(im)[:-4]
                    + "_{:03d}.tif".format(j),
                    center_image_crop,
                )


def process_one_hot(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size,
    center,
    one_hot=True,
    norm="min-max-percentile",
    data_type="8-bit",
    normalization_factor=None,
    fraction_max_ids=1.0,
    rle_encode=False,
):
    """Entry function which generates 2D crops from 2D images (one_hot is True)

    Parameters
    ----------
    im: numpy array
        raw image
    inst: numpy array
        label mask
    crops_dir: str
        Path to where the crops are saved
    data_subset: str
        Set equal to one of `train` or `val`
    crop_size: int
        Height and Width of crop
    center: str
        Set equal to one of `centroid`, `medoid` or `approximate-medoid`
    one_hot: bool
        If label masks are available in one-hot encoded fashion,
        set `one_hot` equal to True
    norm: str
        Set equal to one of `min-max-percentile` or `absolute` or `mean-std`
    data_type: str
        Set equal to `8-bit` or `16-bit`
    normalization_factor: int
        In case, norm is set equal to `absolute`,
        then the image intensities are divided by the `normalization_factor`
    fraction_max_ids: float, between 0 and 1.0
        If set equal to a value less than 1.0,
        then only that fraction of ids are processed are used to make crops
    rle_encode: bool
        If set equal to True, the label masks are saved as csv files
    Returns
    -------
    """

    image_path = os.path.join(crops_dir, data_subset, "images/")
    instance_path = os.path.join(crops_dir, data_subset, "masks/")
    center_image_path = os.path.join(crops_dir, data_subset, "center-" + center + "/")

    try:
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
        os.makedirs(os.path.dirname(center_image_path))
    except FileExistsError:
        pass

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(float)

    if norm == "min-max-percentile":
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))
    elif norm == "mean-std":
        image = normalize_mean_std(image)
    elif norm == "absolute":
        if data_type == "8-bit":
            if normalization_factor is None:
                image /= 255
            else:
                image /= normalization_factor
        elif data_type == "16-bit":
            if normalization_factor is None:
                image /= 65535
            else:
                image /= normalization_factor
    instance = fill_label_holes(instance)

    h, w = image.shape
    instance_np = np.array(instance, copy=False)

    ids = np.arange(instance.shape[0])

    # loop over instances
    ids_subset = np.random.choice(ids, int(fraction_max_ids * len(ids)), replace=False)
    for j, id in enumerate(ids_subset):
        y, x = np.where(instance_np[id] == 1)
        ym, xm = np.mean(y), np.mean(x)

        jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
        ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

        if (
            image[jj : jj + crop_size, ii : ii + crop_size].shape
            == (crop_size, crop_size)
            and np.sum(instance_np[id, jj : jj + crop_size, ii : ii + crop_size]) > 0
        ):
            im_crop = image[jj : jj + crop_size, ii : ii + crop_size]
            instance_crop = instance_np[:, jj : jj + crop_size, ii : ii + crop_size]
            center_image_crop = generate_center_image(
                instance_crop, center, ids, one_hot
            )
            instance_crop_sparse = sparsify(instance_crop)
            tifffile.imsave(
                image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                im_crop,
            )
            if rle_encode:
                encode(
                    instance_path + os.path.basename(im)[:-4] + "_{:03d}.csv".format(j),
                    instance_crop_sparse,
                    one_hot=True,
                )
                encode(
                    center_image_path
                    + os.path.basename(im)[:-4]
                    + "_{:03d}.csv".format(j),
                    center_image_crop,
                    one_hot=False,
                )
            else:
                tifffile.imsave(
                    instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j),
                    instance_crop_sparse,
                )
                tifffile.imsave(
                    center_image_path
                    + os.path.basename(im)[:-4]
                    + "_{:03d}.tif".format(j),
                    center_image_crop,
                )


def process_3d_sliced(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size_x,
    crop_size_y,
    crop_size_z,
    center,
    one_hot=False,
    anisotropy_factor=1.0,
    norm="min-max-percentile",
    data_type="8-bit",
    normalization_factor=None,
    fraction_max_ids=0.01,
    rle_encode=False,
    background_id=0,
):
    """
    Processes the actual images and corresponding GT label masks
    to generate crops of size `crop-size`.
    Here each volumetric (3D) image is processed sliced-wise
    to generate 2D crops in XY, YZ and ZX dimensions.

    Parameters
    ----------
    im: numpy array
        Raw image which must be processed (segmented)
    inst: numpy array
        Corresponding instance mask which contains objects
        present in a one-hot encoded style (DYX)
    crops_dir: string
        Indicates the path where the crops should be saved
    crop_size_x: int
        Length of the crop along the x dimension
    crop_size_y: int
        Length of the crop along the y dimension
    crop_size_z: int
        Length of the crop along the z dimension
    center: string,
        One of `centroid`, `approximate-medoid` or `medoid`
    one_hot: boolean
        If True,  each object is encoded as one in it own individual slice
        in the `inst` image and zero elsewhere.
        False, by default
    anisotropy_factor: float
            This parameter should be set equal to the ratio
            of the z pixel size to the x or y pixel size
            Here, we assume that the x or y pixel size is the same
            (If the imaging is downsampled in the z dimension,
            `anisotropy_factor` is greater than 1.0)
    norm: str
        This should be set equal to one of 'min-max-percentile',
        'mean-std' or 'absolute'
    data_type: str
        This should be set equal to one of '8-bit' or '16-bit'
        This only comes into play if norm is set equal to 'absolute'
    normalization_factor: int, None by default
        This can be set equal to any arbitrary value
        If not set, then for '8-bit' images, it is taken as 255
        and for '16-bit' images, it is taken as 65535
        This only comes into play if norm is set equal to 'absolute'
    fraction_max_ids: float, between 0 and 1.0
        This can be set to a small value in case one doesn't wish
        to produce 2D object centred crops from all objects
        but only a fraction of objects
    rle_encode: bool
        If true, then label masks for crops are not saved as tiffs
        but as csv files (in a run-length encoded fashion)
    background_id: int
        The label id which corresponds to the background
        This is sset equal to 0 by default
    Returns
        -------
    """

    image_path = os.path.join(crops_dir, data_subset, "images/")
    instance_path = os.path.join(crops_dir, data_subset, "masks/")
    center_image_path = os.path.join(crops_dir, data_subset, "center-" + center + "/")

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))
    if not os.path.exists(center_image_path):
        os.makedirs(os.path.dirname(center_image_path))
        print("Created new directory : {}".format(center_image_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(float)
    if norm == "min-max-percentile":
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
    elif norm == "mean-std":
        image = normalize_mean_std(image)
    elif norm == "absolute":
        if data_type == "8-bit":
            if normalization_factor is None:
                image /= 255
            else:
                image /= normalization_factor
        elif data_type == "16-bit":
            if normalization_factor is None:
                image /= 65535
            else:
                image /= normalization_factor
    instance = fill_label_holes(instance)

    # upsample
    image = zoom(image, (anisotropy_factor, 1, 1), order=0)
    instance = zoom(instance, (anisotropy_factor, 1, 1), order=0)

    d, h, w = image.shape
    instance_np = np.array(instance, copy=False)

    object_mask = instance_np > background_id
    # ensure that background is mapped to 0
    instance_np[instance_np == background_id] = 0
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]

    # loop over instances
    ids_subset = np.random.choice(
        ids, int(np.ceil(fraction_max_ids * len(ids))), replace=False
    )
    for id in ids_subset:
        z, y, x = np.where(instance_np == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
        jj = int(np.clip(ym - crop_size_y / 2, 0, h - crop_size_y))
        ii = int(np.clip(xm - crop_size_x / 2, 0, w - crop_size_x))

        # ZY
        for x in range(ii, ii + crop_size_x, int(np.ceil(anisotropy_factor))):
            if image[kk : kk + crop_size_z, jj : jj + crop_size_y, x].shape == (
                crop_size_z,
                crop_size_y,
            ):
                im_crop = image[kk : kk + crop_size_z, jj : jj + crop_size_y, x]
                instance_crop = instance_np[
                    kk : kk + crop_size_z, jj : jj + crop_size_y, x
                ]
                if np.sum(instance_crop) > 0:  # shouldn't be only background
                    center_image_crop = generate_center_image(
                        instance_crop, center, ids, one_hot=one_hot
                    )
                    tifffile.imsave(
                        image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}".format(id)
                        + "_{:03d}".format(x)
                        + "_ZY.tif",
                        im_crop,
                    )
                    if rle_encode:
                        encode(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(x)
                            + "_ZY.csv",
                            instance_crop.astype(np.uint16),
                        )
                        encode(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(x)
                            + "_ZY.csv",
                            center_image_crop,
                        )
                    else:
                        tifffile.imsave(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(x)
                            + "_ZY.tif",
                            instance_crop.astype(np.uint16),
                        )
                        tifffile.imsave(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(x)
                            + "_ZY.tif",
                            center_image_crop,
                        )

        # YX
        for z in range(kk, kk + crop_size_z, int(np.ceil(anisotropy_factor))):
            if image[z, jj : jj + crop_size_y, ii : ii + crop_size_x].shape == (
                crop_size_y,
                crop_size_x,
            ):
                im_crop = image[z, jj : jj + crop_size_y, ii : ii + crop_size_x]
                instance_crop = instance_np[
                    z, jj : jj + crop_size_y, ii : ii + crop_size_x
                ]
                if np.sum(instance_crop) > 0:
                    center_image_crop = generate_center_image(
                        instance_crop, center, ids, one_hot=one_hot
                    )
                    tifffile.imsave(
                        image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}".format(id)
                        + "_{:03d}".format(z)
                        + "_YX.tif",
                        im_crop,
                    )
                    if rle_encode:
                        encode(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.csv",
                            instance_crop.astype(np.uint16),
                        )
                        encode(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.csv",
                            center_image_crop,
                        )
                    else:
                        tifffile.imsave(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.tif",
                            instance_crop.astype(np.uint16),
                        )
                        tifffile.imsave(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.tif",
                            center_image_crop,
                        )

        # XZ
        for y in range(jj, jj + crop_size_y, int(np.ceil(anisotropy_factor))):
            if image[kk : kk + crop_size_z, y, ii : ii + crop_size_x].shape == (
                crop_size_z,
                crop_size_x,
            ):
                im_crop = np.transpose(
                    image[kk : kk + crop_size_z, y, ii : ii + crop_size_x]
                )
                instance_crop = np.transpose(
                    instance_np[kk : kk + crop_size_z, y, ii : ii + crop_size_x]
                )
                if np.sum(instance_crop) > 0:
                    center_image_crop = generate_center_image(
                        instance_crop, center, ids, one_hot=one_hot
                    )
                    tifffile.imsave(
                        image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}".format(id)
                        + "_{:03d}".format(y)
                        + "_XZ.tif",
                        im_crop,
                    )
                    if rle_encode:
                        encode(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(y)
                            + "_XZ.csv",
                            instance_crop.astype(np.uint16),
                        )
                        encode(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(y)
                            + "_XZ.csv",
                            center_image_crop,
                        )
                    else:
                        tifffile.imsave(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(y)
                            + "_XZ.tif",
                            instance_crop.astype(np.uint16),
                        )
                        tifffile.imsave(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(y)
                            + "_XZ.tif",
                            center_image_crop,
                        )


def process_3d_ilp(
    im,
    inst,
    crops_dir,
    data_subset,
    crop_size_x,
    crop_size_y,
    center,
    one_hot=False,
    norm="min-max-percentile",
    data_type="8-bit",
    normalization_factor=None,
    fraction_max_ids=0.10,
    rle_encode=False,
    background_id=0,
):
    """
    Processes the actual images and corresponding GT label masks
    to generate crops of size `crop-size`.
    Here each volumetric (3D) image is processed sliced-wise
    to generate 2D crops in XY dimensions.

    Parameters
    ----------
    im: numpy array
        Raw image which must be processed (segmented)
    inst: numpy array
        Corresponding instance mask which contains objects present
        in a one-hot encoded style (DYX)
    crops_dir: string
        Indicates the path where the crops should be saved
    crop_size_y: int
        Height of the crop in the y dimension
    crop_size_x: int
        Width of the crop in the x dimension
    center: string,
        One of `centroid`, `approximate-medoid` or `medoid`
    norm: boolean
        Setting this to True would perfom min-max normalization
        on the raw image prior to cropping them
        False by default
    one_hot: boolean
        If True,  each object is encoded as one in it own individual slice
        in the `inst` image and zero elsewhere.
        False, by default
    data_type: str
        This should be set equal to one of '8-bit' or '16-bit'
        This only comes into play if norm is set equal to 'absolute'
    normalization_factor: int, None by default
        This can be set equal to any arbitrary value
        If not set, then for '8-bit' images, it is taken as 255
        and for '16-bit' images, it is taken as 65535
        This only comes into play if norm is set equal to 'absolute'
    fraction_max_ids: float, between 0 and 1.0
        This can be set to a small value in case one doesn't wish to produce
        2D object centred crops from all objects
        but only a fraction of objects
    rle_encode: bool
        If true, then label masks for crops are not saved as tiffs
        but as csv files (in a run-length encoded fashion)
    background_id: int
        The label id which corresponds to the background
        This is sset equal to 0 by default
    Returns
        -------
    """

    image_path = os.path.join(crops_dir, data_subset, "images/")
    instance_path = os.path.join(crops_dir, data_subset, "masks/")
    center_image_path = os.path.join(crops_dir, data_subset, "center-" + center + "/")

    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path))
        print("Created new directory : {}".format(image_path))
    if not os.path.exists(instance_path):
        os.makedirs(os.path.dirname(instance_path))
        print("Created new directory : {}".format(instance_path))
    if not os.path.exists(center_image_path):
        os.makedirs(os.path.dirname(center_image_path))
        print("Created new directory : {}".format(center_image_path))

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im).astype(float)
    if norm == "min-max-percentile":
        image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1, 2))
    elif norm == "mean-std":
        image = normalize_mean_std(image)
    elif norm == "absolute":
        if data_type == "8-bit":
            if normalization_factor is None:
                image /= 255
            else:
                image /= normalization_factor
        elif data_type == "16-bit":
            if normalization_factor is None:
                image /= 65535
            else:
                image /= normalization_factor
    instance = fill_label_holes(instance)

    d, h, w = image.shape
    instance_np = np.array(instance, copy=False)

    for z in tqdm(range(image.shape[0]), position=0, leave=True):
        object_mask = instance_np[z] > background_id
        # ensure that background is mapped to 0
        instance_np[z][instance_np[z] == background_id] = 0
        ids = np.unique(instance_np[z][object_mask])
        ids = ids[ids != 0]
        ids_subset = np.random.choice(
            ids, int(fraction_max_ids * len(ids)), replace=False
        )
        for id in ids_subset:
            y, x = np.where(instance_np[z] == id)
            ym, xm = np.mean(y), np.mean(x)
            jj = int(np.clip(ym - crop_size_y / 2, 0, h - crop_size_y))
            ii = int(np.clip(xm - crop_size_x / 2, 0, w - crop_size_x))
            if image[z, jj : jj + crop_size_y, ii : ii + crop_size_x].shape == (
                crop_size_y,
                crop_size_x,
            ):
                im_crop = image[z, jj : jj + crop_size_y, ii : ii + crop_size_x]
                instance_crop = instance_np[
                    z, jj : jj + crop_size_y, ii : ii + crop_size_x
                ]
                if np.sum(instance_crop) > 0:
                    center_image_crop = generate_center_image(
                        instance_crop, center, ids, one_hot=one_hot
                    )
                    tifffile.imsave(
                        image_path
                        + os.path.basename(im)[:-4]
                        + "_{:03d}".format(id)
                        + "_{:03d}".format(z)
                        + "_YX.tif",
                        im_crop,
                    )
                    if rle_encode:
                        encode(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.csv",
                            instance_crop.astype(np.uint16),
                        )
                        encode(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.csv",
                            center_image_crop,
                        )
                    else:
                        tifffile.imsave(
                            instance_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.tif",
                            instance_crop.astype(np.uint16),
                        )
                        tifffile.imsave(
                            center_image_path
                            + os.path.basename(im)[:-4]
                            + "_{:03d}".format(id)
                            + "_{:03d}".format(z)
                            + "_YX.tif",
                            center_image_crop,
                        )
