import os
import numpy as np
import tifffile
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import distance_matrix
from numba import jit

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img == l
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """
        Fill small holes in label image.
    """

    def grow(sl, interior):
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None: continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
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


@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32) 
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
        Generates a `center_image` which is one (True) for all center locations and zero (False) otherwise.
        Parameters
        ----------
        instance: numpy array
            `instance` image containing unique `ids` for each object (YX)
             or present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
        center: string
            One of 'centroid', 'approximate-medoid' or 'medoid'.
        ids: list
            Unique ids corresponding to the objects present in the instance image.
        one_hot: boolean
            True (in this case, `instance` has shape DYX) or False (in this case, `instance` has shape YX).
    """

    if (not one_hot):
        center_image = np.zeros(instance.shape, dtype=bool)
    else:
        center_image = np.zeros((instance.shape[-2], instance.shape[-1]), dtype=bool)
    for j, id in enumerate(ids):
        if (not one_hot):
            y, x = np.where(instance == id)
        else:
            y, x = np.where(instance[id] == 1)
        if len(y) != 0 and len(x) != 0:
            if (center == 'centroid'):
                ym, xm = np.mean(y), np.mean(x)
            elif (center == 'approximate-medoid'):
                ym_temp, xm_temp = np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
                ym, xm = y[imin], x[imin]
            elif (center == 'medoid'):
                ### option - 1 (scipy `distance_matrix`) (slow-ish)
                #dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((x, y)).transpose())
                #imin = np.argmin(np.sum(dist_matrix, axis=0))
                #ym, xm = y[imin], x[imin]
                
                ### option - 2 (`hdmedoid`) (slightly faster than scipy `distance_matrix`)
                #ym, xm = hd.medoid(np.vstack((y,x))) 
                
                ### option - 3 (`numba`) 
                dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]		
            center_image[int(np.round(ym)), int(np.round(xm))] = True
    return center_image


def generate_center_image_3d(instance, center, ids, one_hot, anisotropy_factor, speed_up):
    center_image = np.zeros(instance.shape, dtype=bool)
    instance_downsampled = instance[:, ::int(speed_up), ::int(speed_up)]  # down sample in x and y
    for j, id in enumerate(ids):
        z, y, x = np.where(instance_downsampled == id)
        if len(y) != 0 and len(x) != 0:
            if (center == 'centroid'):
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
            elif (center == 'approximate-medoid'):
                zm_temp, ym_temp, xm_temp = np.median(z), np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2 + (anisotropy_factor* (z - zm_temp)) ** 2)
                zm, ym, xm = z[imin], y[imin], x[imin]
            elif (center == 'medoid'):
                dist_matrix = pairwise_python(np.vstack((speed_up*x, speed_up*y, anisotropy_factor*z)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                zm, ym, xm = z[imin], y[imin], x[imin]
            center_image[int(np.round(zm)), int(np.round(speed_up*ym)), int(np.round(speed_up*xm))] = True
    return center_image


def sparsify(instance):
    """
        Ensures that no slice in the `instance` image crop is empty (completely zero)
        Parameters
        ----------
        instance: numpy array,
            `instance` image present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
    """
    instance_sparse = []
    for z in range(instance.shape[0]):
        if (np.sum(instance[z, ...]) > 0):
            instance_sparse.append(instance[z, ...])
    return np.array(instance_sparse)


def process(im, inst, crops_dir, data_subset, crop_size, center, norm=False, one_hot=False):
    """
        Processes the actual images and instances to generate crops of size `crop-size`.
        Additionally, one could perform min-max normalization of the crops at this stage (False, by default)
        Parameters
        ----------
        im: numpy array
            Raw image which must be processed (segmented)
        inst: numpy array
            Corresponding instance mask which contains objects identified by their unique ids (YX)
        crops_dir: string
            Indicates the path where the crops should be saved
        center: string
            One of `centroid`, `approximate-medoid` or `medoid`
        norm: boolean
            Setting this to True would perfom min-max normalization on the raw image prior to cropping them
            False by default
        one_hot: boolean
            If True,  each object is encoded as one in it own individual slice in the `inst` image and zero elsewhere.
            False, by default

    """
    image_path = os.path.join(crops_dir, data_subset, 'images/')
    instance_path = os.path.join(crops_dir, data_subset, 'masks/')
    center_image_path = os.path.join(crops_dir, data_subset, 'center-' + center + '/')

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
    image = tifffile.imread(im)

    if (norm):
        image = normalize(image, 1, 99.8, axis=(0, 1))
    instance = fill_label_holes(instance)

    h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = instance_np > 0

    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]

    # loop over instances
    for j, id in enumerate(ids):
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)

        jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
        ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

        if (image[jj:jj + crop_size, ii:ii + crop_size].shape == (crop_size, crop_size)):
            im_crop = image[jj:jj + crop_size, ii:ii + crop_size]
            instance_crop = instance[jj:jj + crop_size, ii:ii + crop_size]
            center_image_crop = generate_center_image(instance_crop, center, ids, one_hot)
            tifffile.imsave(image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), im_crop)
            tifffile.imsave(instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), instance_crop)
            tifffile.imsave(center_image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), center_image_crop)

def process_3d(im, inst, crops_dir, data_subset, crop_size_x, crop_size_y, crop_size_z, center, norm=False, one_hot=False, anisotropy_factor = 1.0, speed_up=1.0):
    image_path = os.path.join(crops_dir, data_subset, 'images/')
    instance_path = os.path.join(crops_dir, data_subset, 'masks/')
    center_image_path = os.path.join(crops_dir, data_subset, 'center-' + center + '/')

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
    image = tifffile.imread(im)

    if (norm):
        image = normalize(image, 1, 99.8, axis=(0, 1))
    instance = fill_label_holes(instance)

    d, h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = instance_np > 0

    ids = np.unique(instance_np[object_mask])
    ids = ids[ids != 0]

    # loop over instances
    for j, id in enumerate(ids):
        z, y, x = np.where(instance_np == id)
        zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
        kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
        jj = int(np.clip(ym - crop_size_y / 2, 0, h - crop_size_y))
        ii = int(np.clip(xm - crop_size_x / 2, 0, w - crop_size_x))

        if (image[kk:kk+crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x].shape == (crop_size_z, crop_size_y, crop_size_x)):
            im_crop = image[kk:kk+crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x]
            instance_crop = instance[kk:kk+crop_size_z, jj:jj + crop_size_y, ii:ii + crop_size_x]
            center_image_crop = generate_center_image_3d(instance_crop, center, ids, one_hot=one_hot, anisotropy_factor=anisotropy_factor, speed_up=speed_up)
            tifffile.imsave(image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), im_crop)
            tifffile.imsave(instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), instance_crop.astype(np.uint16))
            tifffile.imsave(center_image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), center_image_crop)



def process_one_hot(im, inst, cropsDir, dataSubset, crop_size, center, norm=False, one_hot=True):
    """
        Processes the actual images and the one-hot encoded instances to generate crops of size `crop-size`.
        Additionally, one could perform min-max normalization of the crops at this stage (False, by default)
        Parameters
        ----------
        im: numpy array
            Raw image which must be processed (segmented)
        inst: numpy array
            Corresponding instance mask which contains objects present in a one-hot encoded style (DYX)
        crops_dir: string
            Indicates the path where the crops should be saved
        center: string,
            One of `centroid`, `approximate-medoid` or `medoid`
        norm: boolean
            Setting this to True would perfom min-max normalization on the raw image prior to cropping them
            False by default
        one_hot: boolean
            If True,  each object is encoded as one in it own individual slice in the `inst` image and zero elsewhere.
            False, by default

    """

    image_path = os.path.join(cropsDir, dataSubset, 'images/')
    instance_path = os.path.join(cropsDir, dataSubset, 'masks/')
    center_image_path = os.path.join(cropsDir, dataSubset, 'center-' + center + '/')

    try:
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
        os.makedirs(os.path.dirname(center_image_path))
    except FileExistsError:
        pass

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im)  # TODO

    if (norm):
        image = normalize(image, 1, 99.8, axis=(0, 1))
    instance = fill_label_holes(instance)
    
    h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = instance_np > 0

    ids = np.arange(instance.shape[0])

    # loop over instances
    for j, id in enumerate(ids):

        y, x = np.where(instance_np[id] == 1)
        ym, xm = np.mean(y), np.mean(x)

        jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
        ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

        if (image[jj:jj + crop_size, ii:ii + crop_size].shape == (crop_size, crop_size) and np.sum(
                instance_np[id, jj:jj + crop_size, ii:ii + crop_size]) > 0):
            im_crop = image[jj:jj + crop_size, ii:ii + crop_size]
            instance_crop = instance[:, jj:jj + crop_size, ii:ii + crop_size]
            center_image_crop = generate_center_image(instance_crop, center, ids, one_hot)
            instance_crop_sparse = sparsify(instance_crop)
            tifffile.imsave(image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), im_crop)  # TODO
            tifffile.imsave(instance_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), instance_crop_sparse)
            tifffile.imsave(center_image_path + os.path.basename(im)[:-4] + "_{:03d}.tif".format(j), center_image_crop)
