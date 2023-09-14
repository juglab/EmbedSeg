import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.metrics import matching_dataset, obtain_APdsb_one_hot
from EmbedSeg.utils.utils import Cluster, Cluster_3d
import numpy as np
from tifffile import imsave
from EmbedSeg.utils.test_time_augmentation import apply_tta_2d, apply_tta_3d
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar, linear_sum_assignment
from skimage.segmentation import relabel_sequential

torch.backends.cudnn.benchmark = True


def begin_evaluating(
    test_configs, optimize=False, maxiter=10, verbose=False, mask_region=None
):
    """Entry function for inferring on test images

    Parameters
    ----------
    test_configs : dictionary
        Dictionary containing testing-specific parameters
        (for e.g. the `seed_thresh`  to use)
    optimize : bool, optional
        It is possible to determine the best performing `fg_thresh`
        by optimizing over different values on the validation sub-set
        By default and in the absence of optimization (i.e. `optimize=False`),
        the fg_thresh  is set equal to 0.5
    maxiter: int
        Number of iterations of optimization.
        Comes into play, only if `optimize=True`
    verbose: bool, optional
        If set equal to True, prints the AP_dsb for each image individually
    mask_region: list of lists, optional
        If a certain region of the image != labelled in the GT label mask,
        that can be specified here.
        This enables comparison of the model prediction only
        with the area which is labeled in the GT label mask
    Returns
    -------
    result_dic: Dictionary
        Keys include the employed `fg_thresh` and the corresponding `AP_dsb`
        at IoU threshold = 0.5
    """
    n_sigma = test_configs["n_sigma"]
    ap_val = test_configs["ap_val"]
    min_mask_sum = test_configs["min_mask_sum"]
    min_unclustered_sum = test_configs["min_unclustered_sum"]
    min_object_size = test_configs["min_object_size"]
    mean_object_size = test_configs["mean_object_size"]
    tta = test_configs["tta"]
    seed_thresh = test_configs["seed_thresh"]
    fg_thresh = test_configs["fg_thresh"]
    save_images = test_configs["save_images"]
    save_results = test_configs["save_results"]
    save_dir = test_configs["save_dir"]
    anisotropy_factor = test_configs["anisotropy_factor"]
    grid_x = test_configs["grid_x"]
    grid_y = test_configs["grid_y"]
    grid_z = test_configs["grid_z"]
    pixel_x = test_configs["pixel_x"]
    pixel_y = test_configs["pixel_y"]
    pixel_z = test_configs["pixel_z"]
    one_hot = test_configs["dataset"]["kwargs"]["one_hot"]
    cluster_fast = test_configs["cluster_fast"]
    expand_grid = test_configs["expand_grid"]
    uniform_ds_factor = test_configs["uniform_ds_factor"]

    # set device
    device = torch.device(test_configs["device"])

    # dataloader
    dataset = get_dataset(
        test_configs["dataset"]["name"], test_configs["dataset"]["kwargs"]
    )
    dataset_it = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=test_configs["num_workers"],
        pin_memory=True if test_configs["device"][:4] == "cuda" else False,
    )

    # load model
    model = get_model(test_configs["model"]["name"], test_configs["model"]["kwargs"])
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)

    # load snapshot
    if os.path.exists(test_configs["checkpoint_path"]):
        state = torch.load(test_configs["checkpoint_path"], map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)  # TODO
    else:
        assert False, "checkpoint_path {} does not exist!".format(
            test_configs["checkpoint_path"]
        )

    # test on evaluation images:
    result_dic = {}
    if test_configs["name"] == "2d":
        args = (
            seed_thresh,
            ap_val,
            min_mask_sum,
            min_unclustered_sum,
            min_object_size,
            tta,
            model,
            dataset_it,
            save_images,
            save_results,
            save_dir,
            verbose,
            grid_x,
            grid_y,
            pixel_x,
            pixel_y,
            one_hot,
            n_sigma,
            cluster_fast,
            expand_grid,
            device,
        )
        if optimize:
            result = minimize_scalar(
                fun=test,
                bounds=(0.3, 0.7),
                method="bounded",
                args=args,
                options={"maxiter": maxiter},
            )
            result_dic["seed_thresh"] = seed_thresh
            result_dic["fg_thresh"] = result.x
            result_dic["AP_dsb_05"] = -result.fun
            print(
                "Optimal fg thresh parameter calculated equal to {:.05f}".format(
                    result.x
                )
            )
            print(
                "AP_dsb at IoU threshold = {} with \
                        fg_thresh = {:.05f} equals {:.05f}".format(
                    0.50, result.x, -result.fun
                )
            )
        else:
            result = test(fg_thresh, *args)
            result_dic["fg_thresh"] = fg_thresh
            result_dic["AP_dsb_05"] = -result
    elif test_configs["name"] == "3d":
        args = (
            seed_thresh,
            ap_val,
            min_mask_sum,
            min_unclustered_sum,
            min_object_size,
            tta,
            model,
            dataset_it,
            save_images,
            save_results,
            save_dir,
            verbose,
            grid_x,
            grid_y,
            grid_z,
            pixel_x,
            pixel_y,
            pixel_z,
            one_hot,
            mask_region,
            n_sigma,
            cluster_fast,
            expand_grid,
            uniform_ds_factor,
            device,
        )

        if optimize:
            result = minimize_scalar(
                fun=test_3d,
                bounds=(0.3, 0.7),
                method="bounded",
                args=args,
                options={"maxiter": maxiter},
            )
            result_dic["seed_thresh"] = seed_thresh
            result_dic["fg_thresh"] = result.x
            result_dic["AP_dsb_05"] = -result.fun
            print(
                "Optimal fg thresh parameter calculated equal to {:.05f}".format(
                    result.x
                )
            )
            print(
                "AP_dsb at IoU threshold = {} with \
                        fg_thresh = {:.05f} equals {:.05f}".format(
                    0.50, result.x, -result.fun
                )
            )
        else:
            result = test_3d(fg_thresh, *args)
            result_dic["fg_thresh"] = fg_thresh
            result_dic["AP_dsb_05"] = -result
    elif test_configs["name"] == "3d_sliced":
        args = (
            seed_thresh,
            ap_val,
            min_mask_sum,
            min_unclustered_sum,
            min_object_size,
            tta,
            model,
            dataset_it,
            save_images,
            save_results,
            save_dir,
            verbose,
            grid_x,
            grid_y,
            grid_z,
            pixel_x,
            pixel_y,
            pixel_z,
            one_hot,
            mask_region,
            n_sigma,
            anisotropy_factor,
            device,
        )
        if optimize:
            result = minimize_scalar(
                fun=test_3d_sliced,
                bounds=(0.3, 0.7),
                method="bounded",
                args=args,
                options={"maxiter": maxiter},
            )
            result_dic["seed_thresh"] = seed_thresh
            result_dic["fg_thresh"] = result.x
            result_dic["AP_dsb_05"] = -result.fun
        else:
            result = test_3d_sliced(fg_thresh, *args)
            result_dic["fg_thresh"] = fg_thresh
            result_dic["AP_dsb_05"] = -result
    elif test_configs["name"] == "3d_ilp":
        args = (
            seed_thresh,
            ap_val,
            min_mask_sum,
            min_unclustered_sum,
            min_object_size,
            mean_object_size,
            tta,
            model,
            dataset_it,
            save_images,
            save_results,
            save_dir,
            verbose,
            grid_x,
            grid_y,
            grid_z,
            pixel_x,
            pixel_y,
            pixel_z,
            one_hot,
            mask_region,
            n_sigma,
            anisotropy_factor,
            device,
        )
        if optimize:
            result = minimize_scalar(
                fun=test_3d_ilp,
                bounds=(0.3, 0.7),
                method="bounded",
                args=args,
                options={"maxiter": maxiter},
            )
            result_dic["seed_thresh"] = seed_thresh
            result_dic["fg_thresh"] = result.x
            result_dic["AP_dsb_05"] = -result.fun
        else:
            result = test_3d_ilp(fg_thresh, *args)
            result_dic["fg_thresh"] = fg_thresh
            result_dic["AP_dsb_05"] = -result
    return result_dic


def stitch_2d(
    instance_map_tile,
    instance_map_current,
    y_current=None,
    x_current=None,
    last=1,
    num_overlap_pixels=4,
    device=None,
):
    """
    Stitching instance segmentations together in case the full 2D image
    doesn't fit in one go, on the GPU
    This function is executed only if `expand_grid` is set to False
    The key idea is we identify the unique ids in the instance_map_current
    and the tile, but only in the overlap region.
    Then we look at the IoU of these. If there is more than 50 % IoU,
    then these are considered to be the same
    Else, a new id is generated!

    Parameters
    ----------
    instance_map_tile : numpy array
        instance segmentation over a tiled view of the image

    instance_map_current: numpy array
        instance segmentation over the complete, large image


    y_current: int
        y position of the top left corner of the tile wrt the complete image

    x_current: int
        x position of the top left corner of the tile wrt the complete image

    last: int
        number of objects currently present in the `instance_map_current`

    num_overlap_pixels: int
        number of overlapping pixels while considering the next tile

    Returns
    -------
    tuple (int, numpy array)
        (updated number of objects currently present in the `instance_map_current`,
        updated instance segmentation over the full image)

    """
    if device is None:
        device = torch.device("cpu")

    mask = instance_map_tile > 0  # foreground region, which has been clustered

    h_tile = instance_map_tile.shape[0]
    w_tile = instance_map_tile.shape[1]
    h_current = instance_map_current.shape[0]
    w_current = instance_map_current.shape[1]

    instance_map_tile_sequential = np.zeros_like(instance_map_tile)

    if mask.sum() > 0:  # i.e. there were some object predictions
        # make sure that instance_map_tile is labeled sequentially

        ids, _, _ = relabel_sequential(instance_map_tile[mask])
        instance_map_tile_sequential[mask] = ids
        instance_map_tile = instance_map_tile_sequential

        # next pad the tile so that it is aligned wrt the complete image
        # note that doing the padding ensures that the instance_map_tile
        # is the same size as the instance_map_current

        instance_map_tile = np.pad(
            instance_map_tile,
            (
                (y_current, np.maximum(0, h_current - y_current - h_tile)),
                (x_current, np.maximum(0, w_current - x_current - w_tile)),
            ),
        )

        # ensure that it has the same shape as instance_map_current
        instance_map_tile = instance_map_tile[:h_current, :w_current]

        mask_overlap = np.zeros_like(instance_map_tile)
        # this just identifies the region where the tile
        # overlaps with the `instance_map_current`

        if y_current == 0 and x_current == 0:
            ids_tile = np.unique(instance_map_tile)
            ids_tile = ids_tile[ids_tile != 0]  # ignore background
            instance_map_current[:h_current, :w_current] = instance_map_tile
            last = len(ids_tile) + 1
        else:
            if x_current != 0 and y_current == 0:
                mask_overlap[
                    y_current : y_current + h_tile,
                    x_current : x_current + num_overlap_pixels,
                ] = 1
            elif x_current == 0 and y_current != 0:
                mask_overlap[
                    y_current : y_current + num_overlap_pixels,
                    x_current : x_current + w_tile,
                ] = 1
            elif x_current != 0 and y_current != 0:
                mask_overlap[
                    y_current : y_current + h_tile,
                    x_current : x_current + num_overlap_pixels,
                ] = 1
                mask_overlap[
                    y_current : y_current + num_overlap_pixels,
                    x_current : x_current + w_tile,
                ] = 1

            # identify ids in the complete tile, not just the overlap region,
            ids_tile_all = np.unique(instance_map_tile)
            ids_tile_all = ids_tile_all[ids_tile_all != 0]

            # identify ids in the the overlap region,
            ids_tile_overlap = np.unique(instance_map_tile * mask_overlap)
            ids_tile_overlap = ids_tile_overlap[ids_tile_overlap != 0]

            # identify ids not in overlap region
            ids_tile_notin_overlap = np.setdiff1d(ids_tile_all, ids_tile_overlap)

            # identify ids in `instance_map_current` but only in the overlap region
            instance_map_current_masked = torch.from_numpy(
                instance_map_current * mask_overlap
            ).to(device)

            ids_current_overlap = (
                torch.unique(instance_map_current_masked).cpu().detach().numpy()
            )
            ids_current_overlap = ids_current_overlap[ids_current_overlap != 0]

            IoU_table = np.zeros((len(ids_tile_overlap), len(ids_current_overlap)))
            instance_map_tile_masked = torch.from_numpy(
                instance_map_tile * mask_overlap
            ).to(device)

            # rows are ids in tile, cols are ids in GT instance map

            for i, id_tile_overlap in enumerate(ids_tile_overlap):
                for j, id_current_overlap in enumerate(ids_current_overlap):
                    intersection = (
                        (instance_map_tile_masked == id_tile_overlap)
                        & (instance_map_current_masked == id_current_overlap)
                    ).sum()
                    union = (
                        (instance_map_tile_masked == id_tile_overlap)
                        | (instance_map_current_masked == id_current_overlap)
                    ).sum()
                    if union != 0:
                        IoU_table[i, j] = intersection / union
                    else:
                        IoU_table[i, j] = 0.0

            row_indices, col_indices = linear_sum_assignment(-IoU_table)
            matched_indices = np.array(
                list(zip(row_indices, col_indices))
            )  # list of (row, col) tuples
            unmatched_indices_tile = np.setdiff1d(
                np.arange(len(ids_tile_overlap)), row_indices
            )

            for m in matched_indices:
                if IoU_table[m[0], m[1]] >= 0.5:  # (tile, current)
                    # wherever the tile is m[0], it should be assigned
                    # m[1] in the larger prediction image
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = ids_current_overlap[m[1]]
                elif IoU_table[m[0], m[1]] == 0:
                    # there is no intersection
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    last += 1
                else:
                    # otherwise just take a union of the both ...
                    # basically this should spawn a new label, since there
                    # was not a satisfactory match with any pre-existing id
                    # in the instance_map_current
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    # instance_map_current[instance_map_current == \
                    #        ids_current_overlap[m[1]]] = last  # TODO
                    last += 1
            for index in unmatched_indices_tile:  # not a tuple
                instance_map_current[
                    instance_map_tile == ids_tile_overlap[index]
                ] = last
                last += 1
            for id in ids_tile_notin_overlap:
                instance_map_current[instance_map_tile == id] = last
                last += 1
        return last, instance_map_current
    else:
        return (
            last,
            instance_map_current,
        )  # if there are no ids in tile, then just return


def stitch_3d(
    instance_map_tile,
    instance_map_current,
    z_tile=None,
    y_tile=None,
    x_tile=None,
    last=1,
    num_overlap_pixels=4,
    device=None,
):
    """Stitching instance segmentations together in case the
    full 3D image doesn't fit in one go, on the GPU
    This function is executed only if `expand_grid` is set to False

             Parameters
             ----------
             instance_map_tile : numpy array
                 instance segmentation over a tiled view of the image

             instance_map_current: numpy array
                 instance segmentation over the complete, large image

             z_tile: int
                 z position of the top left corner of the tile wrt the complete image

             y_tile: int
                 y position of the top left corner of the tile wrt the complete image

             x_tile: int
                 x position of the top left corner of the tile wrt the complete image

             last: int
                 number of objects currently present in the `instance_map_current`

             num_overlap_pixels: int
                 number of overlapping pixels while considering the next tile

             Returns
             -------
             tuple (int, numpy array)
                 (updated number of objects currently
                  present in the `instance_map_current`,
                 updated instance segmentation over the full image)

    """
    if device is None:
        device = torch.device("cpu")

    mask = instance_map_tile > 0

    D = instance_map_tile.shape[0]
    H = instance_map_tile.shape[1]
    W = instance_map_tile.shape[2]

    instance_map_tile_sequential = np.zeros_like(instance_map_tile)

    if mask.sum() > 0:  # i.e. there were some object predictions
        # make sure that instance_map_tile is labeled sequentially

        ids, _, _ = relabel_sequential(instance_map_tile[mask])
        instance_map_tile_sequential[mask] = ids
        instance_map_tile = instance_map_tile_sequential

        # next pad the tile so that it is aligned wrt the complete image

        instance_map_tile = np.pad(
            instance_map_tile,
            (
                (z_tile, np.maximum(0, instance_map_current.shape[0] - z_tile - D)),
                (y_tile, np.maximum(0, instance_map_current.shape[1] - y_tile - H)),
                (x_tile, np.maximum(0, instance_map_current.shape[2] - x_tile - W)),
            ),
        )

        # ensure that it has the same shape as instance_map_current
        instance_map_tile = instance_map_tile[
            : instance_map_current.shape[0],
            : instance_map_current.shape[1],
            : instance_map_current.shape[2],
        ]
        mask_overlap = np.zeros_like(instance_map_tile)

        if z_tile == 0 and y_tile == 0 and x_tile == 0:
            ids_tile = np.unique(instance_map_tile)
            ids_tile = ids_tile[ids_tile != 0]
            instance_map_current[
                : instance_map_tile.shape[0],
                : instance_map_tile.shape[1],
                : instance_map_tile.shape[2],
            ] = instance_map_tile
            last = len(ids_tile) + 1
        else:
            if x_tile != 0 and y_tile == 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
            elif x_tile == 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile == 0 and y_tile == 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile == 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
            elif x_tile == 0 and y_tile != 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile != 0:
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + H,
                    x_tile : x_tile + num_overlap_pixels,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + D,
                    y_tile : y_tile + num_overlap_pixels,
                    x_tile : x_tile + W,
                ] = 1
                mask_overlap[
                    z_tile : z_tile + num_overlap_pixels,
                    y_tile : y_tile + H,
                    x_tile : x_tile + W,
                ] = 1

            # identify ids in the complete tile, not just the overlap region,
            ids_tile_all = np.unique(instance_map_tile)
            ids_tile_all = ids_tile_all[ids_tile_all != 0]

            # identify ids in the the overlap region,
            ids_tile_overlap = np.unique(instance_map_tile * mask_overlap)
            ids_tile_overlap = ids_tile_overlap[ids_tile_overlap != 0]

            # identify ids not in overlap region
            ids_tile_notin_overlap = np.setdiff1d(ids_tile_all, ids_tile_overlap)

            # identify ids in `instance_map_current` but only in the overlap region
            instance_map_current_masked = torch.from_numpy(
                instance_map_current * mask_overlap
            ).to(device)

            ids_current = (
                torch.unique(instance_map_current_masked).cpu().detach().numpy()
            )
            ids_current = ids_current[ids_current != 0]

            IoU_table = np.zeros((len(ids_tile_overlap), len(ids_current)))
            instance_map_tile_masked = torch.from_numpy(
                instance_map_tile * mask_overlap
            ).to(device)

            # rows are ids in tile, cols are ids in GT instance map

            for i, id_tile in enumerate(ids_tile_overlap):
                for j, id_current in enumerate(ids_current):
                    intersection = (
                        (instance_map_tile_masked == id_tile)
                        & (instance_map_current_masked == id_current)
                    ).sum()
                    union = (
                        (instance_map_tile_masked == id_tile)
                        | (instance_map_current_masked == id_current)
                    ).sum()
                    if union != 0:
                        IoU_table[i, j] = intersection / union
                    else:
                        IoU_table[i, j] = 0.0

            row_indices, col_indices = linear_sum_assignment(-IoU_table)
            matched_indices = np.array(
                list(zip(row_indices, col_indices))
            )  # list of (row, col) tuples
            unmatched_indices_tile = np.setdiff1d(
                np.arange(len(ids_tile_overlap)), row_indices
            )

            for m in matched_indices:
                if IoU_table[m[0], m[1]] >= 0.5:  # (tile, current)
                    # wherever the tile is m[0],
                    # it should be assigned m[1] in the larger prediction image
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = ids_current[m[1]]
                elif IoU_table[m[0], m[1]] == 0:
                    # there is no intersection
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    last += 1
                else:
                    # otherwise just take a union of the both ...
                    instance_map_current[
                        instance_map_tile == ids_tile_overlap[m[0]]
                    ] = last
                    # instance_map_current[instance_map_current
                    #                       == ids_current[m[1]]] = last
                    last += 1
            for index in unmatched_indices_tile:  # not a tuple
                instance_map_current[
                    instance_map_tile == ids_tile_overlap[index]
                ] = last
                last += 1
            for id in ids_tile_notin_overlap:
                instance_map_current[instance_map_tile == id] = last
                last += 1

        return last, instance_map_current
    else:
        return (
            last,
            instance_map_current,
        )  # if there are no ids in tile, then just return


def predict(
    im,
    model,
    tta,
    cluster_fast,
    n_sigma,
    fg_thresh,
    seed_thresh,
    min_mask_sum,
    min_unclustered_sum,
    min_object_size,
    cluster,
    device,
):
    """

    Parameters
    ----------
    im : PyTorch Tensor
        BCYX

    model: PyTorch model

    tta: bool
        If True, then Test-Time Augmentation is on, otherwise off
    cluster_fast: bool
        If True, then the cluster.cluster() is used
        If False, then cluster.cluster_local_maxima() is used
    n_sigma: int
        This should be set equal to `2` for a 2D setting
    fg_thresh: float
        This should be set equal to `0.5` by default
    seed_thresh: float
        This should be set equal to `0.9` by default
    min_mask_sum: int
        Only start creating instances,
        if there are at least `min_mask_sum` pixels in foreground!
    min_unclustered_sum: int
        Stop when the number of seed candidates are less than `min_unclustered_sum`
    min_object_size: int
        Predicted Objects below this threshold are ignored

    cluster: Object of class `Cluster`

    Returns
    -------
    instance_map: PyTorch Tensor
        YX
    seed_map: PyTorch Tensor
        YX
    """

    multiple_y = im.shape[2] // 8
    multiple_x = im.shape[3] // 8

    if im.shape[2] % 8 != 0:
        diff_y = 8 * (multiple_y + 1) - im.shape[2]
    else:
        diff_y = 0
    if im.shape[3] % 8 != 0:
        diff_x = 8 * (multiple_x + 1) - im.shape[3]
    else:
        diff_x = 0
    p2d = (
        diff_x // 2,
        diff_x - diff_x // 2,
        diff_y // 2,
        diff_y - diff_y // 2,
    )  # last dim, second last dim

    im = F.pad(im, p2d, "reflect")

    if tta:
        output = apply_tta_2d(im, model)
    else:
        output = model(im)
    if cluster_fast:
        instance_map = cluster.cluster(
            output[0],
            n_sigma=n_sigma,
            seed_thresh=seed_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
    else:
        instance_map = cluster.cluster_local_maxima(
            output[0],
            n_sigma=n_sigma,
            fg_thresh=fg_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )

    seed_map = torch.sigmoid(output[0, -1, ...])
    # unpad instance_map, seed_map
    if (diff_y - diff_y // 2) != 0:
        instance_map = instance_map[diff_y // 2 : -(diff_y - diff_y // 2), ...]
        seed_map = seed_map[diff_y // 2 : -(diff_y - diff_y // 2), ...]
    if (diff_x - diff_x // 2) != 0:
        instance_map = instance_map[..., diff_x // 2 : -(diff_x - diff_x // 2)]
        seed_map = seed_map[..., diff_x // 2 : -(diff_x - diff_x // 2)]
    return instance_map, seed_map


def predict_3d(
    im,
    model,
    tta,
    cluster_fast,
    n_sigma,
    fg_thresh,
    seed_thresh,
    min_mask_sum,
    min_unclustered_sum,
    min_object_size,
    cluster,
    device,
):
    """

    Parameters
    ----------
    im : PyTorch Tensor
        BZCYX

    model: PyTorch model

    tta: bool
        If True, then Test-Time Augmentation is on, otherwise off
    cluster_fast: bool
        If True, then the cluster.cluster() is used
        If False, then cluster.cluster_local_maxima() is used
    n_sigma: int
        This should be set equal to `3` for a 3D setting
    fg_thresh: float
        This should be set equal to `0.5` by default
    seed_thresh: float
        This should be set equal to `0.9` by default
    min_mask_sum: int
        Only start creating instances,
        if there are at least `min_mask_sum` pixels in foreground!
    min_unclustered_sum: int
        Stop when the number of seed candidates are less than `min_unclustered_sum`
    min_object_size: int
        Predicted Objects below this threshold are ignored

    cluster: Object of class `Cluster_3d`

    Returns
    -------
    instance_map: PyTorch Tensor
        ZYX
    seed_map: PyTorch Tensor
        ZYX
    """
    im, diff_x, diff_y, diff_z = pad_3d(im)

    if tta:
        for iter in tqdm(range(16), position=0, leave=True):
            if iter == 0:
                output_average = apply_tta_3d(im, model, iter)
            else:
                output_average = (
                    1
                    / (iter + 1)
                    * (output_average * iter + apply_tta_3d(im, model, iter))
                )  # iter
        output = torch.from_numpy(output_average).float().to(device)
    else:
        output = model(im)

    if cluster_fast:
        instance_map = cluster.cluster(
            output[0],
            n_sigma=n_sigma,
            fg_thresh=fg_thresh,
            seed_thresh=seed_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
    else:
        instance_map = cluster.cluster_local_maxima(
            output[0],
            n_sigma=n_sigma,
            fg_thresh=fg_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
    seed_map = torch.sigmoid(output[0, -1, ...])
    # unpad instance_map, seed_map

    if diff_z != 0:
        instance_map = instance_map[:-diff_z, :, :]
        seed_map = seed_map[:-diff_z, :, :]
    if diff_y != 0:
        instance_map = instance_map[:, :-diff_y, :]
        seed_map = seed_map[:, :-diff_y, :]
    if diff_x != 0:
        instance_map = instance_map[:, :, :-diff_x]
        seed_map = seed_map[:, :, :-diff_x]
    return instance_map, seed_map


def test(fg_thresh, *args):
    """Infer the trained 2D model on 2D images

    Parameters
    ----------
    fg_thresh : float
        foreground threshold decides which pixels are considered for clustering,
        based on the predicted seediness scores at these pixels.
    args: dictionary
        Contains other paremeters such as `ap_val`, `seed_thresh` etc
    Returns
    -------
    float
        Average `AP_dsb` over all test images
    """
    (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        tta,
        model,
        dataset_it,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        pixel_x,
        pixel_y,
        one_hot,
        n_sigma,
        cluster_fast,
        expand_grid,
        device,
    ) = args

    model.eval()

    # cluster module
    cluster = Cluster(grid_y, grid_x, pixel_y, pixel_x, device)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample["image"]  # B 1 Y X
            im = im.to(device)

            H, W = im.shape[2], im.shape[3]
            if H > grid_y or W > grid_x:
                if expand_grid:
                    # simple trick to expand the grid while keeping
                    # pixel resolution the same as before
                    H_, W_ = round_up_8(H), round_up_8(W)
                    temp = np.maximum(H_, W_)
                    H_ = temp
                    W_ = temp
                    pixel_x_modified = pixel_y_modified = H_ / grid_y
                    cluster = Cluster(
                        H_, W_, pixel_y_modified, pixel_x_modified, device
                    )
                    instance_map, seed_map = predict(
                        im,
                        model,
                        tta,
                        cluster_fast,
                        n_sigma,
                        fg_thresh,
                        seed_thresh,
                        min_mask_sum,
                        min_unclustered_sum,
                        min_object_size,
                        cluster,
                        device,
                    )

                else:
                    # here, we try stitching predictions instead
                    last = 1
                    instance_map = np.zeros((H, W), dtype=np.int16)
                    seed_map = np.zeros((H, W), dtype=float)
                    num_overlap_pixels = 4
                    for y in range(0, H, grid_y - num_overlap_pixels):
                        for x in range(0, W, grid_x - num_overlap_pixels):
                            instance_map_tile, seed_map_tile = predict(
                                im[:, :, y : y + grid_y, x : x + grid_x],
                                model,
                                tta,
                                cluster_fast,
                                n_sigma,
                                fg_thresh,
                                seed_thresh,
                                min_mask_sum,
                                min_unclustered_sum,
                                min_object_size,
                                cluster,
                                device,
                            )
                            last, instance_map = stitch_2d(
                                instance_map_tile.cpu().detach().numpy(),
                                instance_map,
                                y,
                                x,
                                last,
                                num_overlap_pixels,
                                device,
                            )
                            seed_map[y : y + grid_y, x : x + grid_x] = (
                                seed_map_tile.cpu().detach().numpy()
                            )
                    instance_map = torch.from_numpy(instance_map).to(device)
                    seed_map = torch.from_numpy(seed_map).float().to(device)
            else:
                instance_map, seed_map = predict(
                    im,
                    model,
                    tta,
                    cluster_fast,
                    n_sigma,
                    fg_thresh,
                    seed_thresh,
                    min_mask_sum,
                    min_unclustered_sum,
                    min_object_size,
                    cluster,
                    device,
                )

            base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
            image_file_names.append(base)

            if one_hot:
                if "instance" in sample:
                    all_results = obtain_APdsb_one_hot(
                        gt_image=sample["instance"].squeeze().cpu().detach().numpy(),
                        prediction_image=instance_map.cpu().detach().numpy(),
                        ap_val=ap_val,
                    )
                    if verbose:
                        print("Accuracy: {:.03f}".format(all_results), flush=True)
                    result_list.append(all_results)
            else:
                if "instance" in sample:
                    all_results = matching_dataset(
                        y_true=[sample["instance"].squeeze().cpu().detach().numpy()],
                        y_pred=[instance_map.cpu().detach().numpy()],
                        thresh=ap_val,
                        show_progress=False,
                    )
                    if verbose:
                        print(
                            "Accuracy: {:.03f}".format(all_results.accuracy), flush=True
                        )
                    result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, "predictions/")):
                    os.makedirs(os.path.join(save_dir, "predictions/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "predictions/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "ground-truth/")):
                    os.makedirs(os.path.join(save_dir, "ground-truth/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "ground-truth/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "seeds/")):
                    os.makedirs(os.path.join(save_dir, "seeds/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "seeds/")
                        )
                    )

                # save predictions
                base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
                instances_file = os.path.join(save_dir, "predictions/", base + ".tif")
                imsave(
                    instances_file,
                    instance_map.cpu().detach().numpy().astype(np.uint16),
                )
                seeds_file = os.path.join(save_dir, "seeds/", base + ".tif")
                imsave(seeds_file, seed_map.cpu().detach().numpy())
                if "instance" in sample:
                    gt_file = os.path.join(save_dir, "ground-truth/", base + ".tif")
                    imsave(gt_file, sample["instance"].squeeze().cpu().detach().numpy())

        if save_results and "instance" in sample:
            if not os.path.exists(os.path.join(save_dir, "results/")):
                os.makedirs(os.path.join(save_dir, "results/"))
                print(
                    "Created new directory {}".format(
                        os.path.join(save_dir, "results/")
                    )
                )
            txt_file = os.path.join(
                save_dir,
                "results/combined_AP-"
                + "{:.02f}".format(ap_val)
                + "_tta-"
                + str(tta)
                + ".txt",
            )
            with open(txt_file, "w") as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, \
                            min_object_size, seed_thresh, \
                            intersection_threshold, accuracy \n"
                )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_tif = im_name + ".tif"
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(
                            im_name_tif,
                            min_mask_sum,
                            min_unclustered_sum,
                            min_object_size,
                            seed_thresh,
                            ap_val,
                            score,
                        )
                    )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines(
                    "Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(
                        ap_val, np.mean(result_list)
                    )
                )

        if len(result_list) != 0:
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold = {} \
                        at foreground threshold = {:.05f}, \
                        is equal to {:.05f}".format(
                    ap_val, fg_thresh, np.mean(result_list)
                )
            )
            return -np.mean(result_list)
        else:
            return 0.0


def round_up_8(x):
    """Helper function for rounding integer to next multiple of 8

    e.g:
    round_up_8(10) = 16

        Parameters
        ----------
        x : int
            Integer
        Returns
        -------
        int
    """
    return (int(x) + 7) & (-8)


def pad_3d(im_tile):
    """Pad a 3D  image so that its dimensions are all multiples of 8

    Parameters
    ----------
    im_tile : numpy array (D x H x W)
       3D Image which needs to be padded!
    Returns
    -------
    (numpy array, int, int, int)
    (Padded 3D image, diff in x, diff in y, diff in z)
    The last three values are the amount of padding needed in the x, y and z dimensions
    """

    multiple_z = im_tile.shape[2] // 8
    multiple_y = im_tile.shape[3] // 8
    multiple_x = im_tile.shape[4] // 8
    if im_tile.shape[2] % 8 != 0:
        diff_z = 8 * (multiple_z + 1) - im_tile.shape[2]
    else:
        diff_z = 0
    if im_tile.shape[3] % 8 != 0:
        diff_y = 8 * (multiple_y + 1) - im_tile.shape[3]
    else:
        diff_y = 0
    if im_tile.shape[4] % 8 != 0:
        diff_x = 8 * (multiple_x + 1) - im_tile.shape[4]
    else:
        diff_x = 0

    p3d = (0, diff_x, 0, diff_y, 0, diff_z)  # last dim, second last dim, third last dim

    im_tile = F.pad(im_tile, p3d, "reflect")
    return im_tile, diff_x, diff_y, diff_z


def test_3d(fg_thresh, *args):
    """Infer the trained 3D model on 3D images

    Parameters
    ----------
    fg_thresh : float
        foreground threshold decides which pixels are considered for clustering,
        based on the predicted seediness scores at these pixels.
    args: dictionary
        Contains other paremeters such as `ap_val`, `seed_thresh` etc
    Returns
    -------
    float
        Average `AP_dsb` over all test images
    """
    (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        tta,
        model,
        dataset_it,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        grid_z,
        pixel_x,
        pixel_y,
        pixel_z,
        one_hot,
        mask_region,
        n_sigma,
        cluster_fast,
        expand_grid,
        uniform_ds_factor,
        device,
    ) = args

    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, device)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample["image"]
            im = im.to(device)

            D, H, W = im.shape[2], im.shape[3], im.shape[4]

            if D > grid_z or H > grid_y or W > grid_x:
                if expand_grid:
                    D_, H_, W_ = round_up_8(D), round_up_8(H), round_up_8(W)
                    temp = np.maximum(H_, W_)
                    H_ = temp
                    W_ = temp
                    pixel_x_modified = pixel_y_modified = H_ / grid_y
                    pixel_z_modified = D_ * pixel_z / grid_z
                    cluster = Cluster_3d(
                        D_,
                        H_,
                        W_,
                        pixel_z_modified,
                        pixel_y_modified,
                        pixel_x_modified,
                        device,
                    )
                    instance_map, seed_map = predict_3d(
                        im,
                        model,
                        tta,
                        cluster_fast,
                        n_sigma,
                        fg_thresh,
                        seed_thresh,
                        min_mask_sum,
                        min_unclustered_sum,
                        min_object_size,
                        cluster,
                        device,
                    )
                else:
                    # here, we try stitching predictions instead
                    last = 1
                    instance_map = np.zeros((D, H, W), dtype=np.int16)
                    seed_map = np.zeros((D, H, W), dtype=float)
                    num_overlap_pixels = 4
                    for z in range(0, D, grid_z - num_overlap_pixels):
                        for y in range(0, H, grid_y - num_overlap_pixels):
                            for x in range(0, W, grid_x - num_overlap_pixels):
                                instance_map_tile, seed_map_tile = predict_3d(
                                    im[
                                        :,
                                        :,
                                        z : z + grid_z,
                                        y : y + grid_y,
                                        x : x + grid_x,
                                    ],
                                    model,
                                    tta,
                                    cluster_fast,
                                    n_sigma,
                                    fg_thresh,
                                    seed_thresh,
                                    min_mask_sum,
                                    min_unclustered_sum,
                                    min_object_size,
                                    cluster,
                                    device,
                                )
                                last, instance_map = stitch_3d(
                                    instance_map_tile.cpu().detach().numpy(),
                                    instance_map,
                                    z,
                                    y,
                                    x,
                                    last,
                                    num_overlap_pixels,
                                    device,
                                )
                                seed_map[
                                    z : z + grid_z, y : y + grid_y, x : x + grid_x
                                ] = (seed_map_tile.cpu().detach().numpy())
                    instance_map = torch.from_numpy(instance_map).to(device)
                    seed_map = torch.from_numpy(seed_map).float().to(device)
            else:
                instance_map, seed_map = predict_3d(
                    im,
                    model,
                    tta,
                    cluster_fast,
                    n_sigma,
                    fg_thresh,
                    seed_thresh,
                    min_mask_sum,
                    min_unclustered_sum,
                    min_object_size,
                    cluster,
                    device,
                )

            # zoom back to original size
            instance_map = zoom(
                instance_map.cpu().detach().numpy().astype(np.uint16),
                uniform_ds_factor,
                order=0,
            )
            seed_map = zoom(seed_map.cpu().detach().numpy(), uniform_ds_factor)

            if "instance" in sample:
                instances = sample["instance"].squeeze()  # Z, Y, X

            if mask_region is not None:
                # ignore predictions in this region prior to saving the tiffs
                # or prior to comparison with GT masks
                instance_map[
                    int(mask_region[0][0]) : int(mask_region[1][0]),
                    int(mask_region[0][1]) : int(mask_region[1][1]),
                    int(mask_region[0][2]) : int(mask_region[1][2]),
                ] = 0  # Z Y X
            else:
                pass

            if "instance" in sample:
                instance_map = instance_map[
                    : instances.shape[0], : instances.shape[1], : instances.shape[2]
                ]
                seed_map = seed_map[
                    : instances.shape[0], : instances.shape[1], : instances.shape[2]
                ]
                all_results = matching_dataset(
                    y_true=[instances.cpu().detach().numpy()],
                    y_pred=[instance_map],
                    thresh=ap_val,
                    show_progress=False,
                )
                if verbose:
                    print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, "predictions/")):
                    os.makedirs(os.path.join(save_dir, "predictions/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "predictions/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "ground-truth/")):
                    os.makedirs(os.path.join(save_dir, "ground-truth/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "ground-truth/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "seeds/")):
                    os.makedirs(os.path.join(save_dir, "seeds/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "seeds/")
                        )
                    )

                base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
                image_file_names.append(base)

                instances_file = os.path.join(save_dir, "predictions/", base + ".tif")
                imsave(instances_file, instance_map)

                seeds_file = os.path.join(save_dir, "seeds/", base + ".tif")
                imsave(seeds_file, seed_map)

                if "instance" in sample:
                    gt_file = os.path.join(save_dir, "ground-truth/", base + ".tif")
                    imsave(gt_file, sample["instance"].squeeze().cpu().detach().numpy())

        if save_results and "instance" in sample:
            if not os.path.exists(os.path.join(save_dir, "results/")):
                os.makedirs(os.path.join(save_dir, "results/"))
                print(
                    "Created new directory {}".format(
                        os.path.join(save_dir, "results/")
                    )
                )
            txt_file = os.path.join(
                save_dir,
                "results/combined_AP-"
                + "{:.02f}".format(ap_val)
                + "_tta-"
                + str(tta)
                + ".txt",
            )
            with open(txt_file, "w") as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, \
                            min_object_size, seed_thresh, \
                            intersection_threshold, accuracy \n"
                )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_tif = im_name + ".tif"
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(
                            im_name_tif,
                            min_mask_sum,
                            min_unclustered_sum,
                            min_object_size,
                            seed_thresh,
                            ap_val,
                            score,
                        )
                    )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines(
                    "Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(
                        ap_val, np.mean(result_list)
                    )
                )

        if len(result_list) != 0:
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold \
                        = {} at seediness threshold = {:.05f}, is equal to \
                        {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)
                )
            )
            return -np.mean(result_list)
        else:
            return 0.0


def test_3d_sliced(fg_thresh, *args):
    """Infer the trained 2D model on 3D images in a `sliced` fashion

    Parameters
    ----------
    fg_thresh : float
        foreground threshold decides which pixels are considered
        for clustering, based on the predicted seediness scores at these pixels.
    args: dictionary
        Contains other paremeters such as `ap_val`, `seed_thresh` etc
    Returns
    -------
    float
        Average `AP_dsb` over all test images
    """
    (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        tta,
        model,
        dataset_it,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        grid_z,
        pixel_x,
        pixel_y,
        pixel_z,
        one_hot,
        mask_region,
        n_sigma,
        anisotropy_factor,
        device,
    ) = args

    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, device)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample["image"]  # isotropically expanded image
            im = im.to(device)
            multiple_z = im.shape[2] // 8
            multiple_y = im.shape[3] // 8
            multiple_x = im.shape[4] // 8

            if im.shape[2] % 8 != 0:
                diff_z = 8 * (multiple_z + 1) - im.shape[2]
            else:
                diff_z = 0
            if im.shape[3] % 8 != 0:
                diff_y = 8 * (multiple_y + 1) - im.shape[3]
            else:
                diff_y = 0
            if im.shape[4] % 8 != 0:
                diff_x = 8 * (multiple_x + 1) - im.shape[4]
            else:
                diff_x = 0
            p3d = (
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            )  # last dim, second last dim, third last dim!

            im = F.pad(im, p3d, "reflect")
            if "instance" in sample:
                instances = sample[
                    "instance"
                ].squeeze()  # isotropically expanded GT instance map

            print("Processing `YX` slices .....")
            # obtain YX slices
            output_YX_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).to(
                device
            )
            for z in tqdm(range(im.shape[2]), position=0, leave=True):
                if tta:
                    output_YX = apply_tta_2d(im[:, :, z, ...], model)
                else:
                    output_YX = model(im[:, :, z, ...])
                output_YX_3d[:, :, z, ...] = output_YX
            print("Processing `ZX` slices .....")
            # obtain ZX slices
            output_ZX_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).to(
                device
            )
            for y in tqdm(range(im.shape[3]), position=0, leave=True):
                if tta:
                    output_ZX = apply_tta_2d(im[:, :, :, y, ...], model)
                else:
                    output_ZX = model(im[:, :, :, y, ...])
                # offset in x, offset in z, sigma in x, sigma in z, seed
                output_ZX_3d[:, :, :, y, ...] = output_ZX
            print("Processing `ZY` slices .....")
            # obtain ZY slices
            output_ZY_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).to(
                device
            )
            for x in tqdm(range(im.shape[4]), position=0, leave=True):
                if tta:
                    output_ZY = apply_tta_2d(im[:, :, :, :, x], model)
                else:
                    output_ZY = model(im[:, :, :, :, x])

                # offset in y, offset in z, sigma in y, sigma in z, seed
                output_ZY_3d[:, :, :, :, x] = output_ZY
            output = (
                torch.from_numpy(
                    np.zeros((1, 7, im.shape[2], im.shape[3], im.shape[4]))
                )
                .float()
                .to(device)
            )  # B (6+1) Z Y X

            # handle seediness
            output[:, 6, ...] = (
                1
                / 3
                * (
                    output_YX_3d[:, 4, ...]
                    + output_ZX_3d[:, 4, ...]
                    + output_ZY_3d[:, 4, ...]
                )
            )

            # handle offset in X
            output[:, 0, ...] = (
                1 / 2 * (output_YX_3d[:, 0, ...] + output_ZX_3d[:, 0, ...])
            )

            # handle offset in Y
            output[:, 1, ...] = (
                1 / 2 * (output_YX_3d[:, 1, ...] + output_ZY_3d[:, 0, ...])
            )

            # handle offset in Z
            output[:, 2, ...] = (
                1 / 2 * (output_ZX_3d[:, 1, ...] + output_ZY_3d[:, 1, ...])
            )

            # sigma in X
            output[:, 3, ...] = (
                1 / 2 * (output_YX_3d[:, 2, ...] + output_ZX_3d[:, 2, ...])
            )

            # sigma in Y
            output[:, 4, ...] = (
                1 / 2 * (output_YX_3d[:, 3, ...] + output_ZY_3d[:, 2, ...])
            )

            # sigma in Z
            output[:, 5, ...] = (
                1 / 2 * (output_ZX_3d[:, 3, ...] + output_ZY_3d[:, 3, ...])
            )

            # unpad output, instances and image
            if (diff_z - diff_z // 2) != 0:
                output = output[:, :, diff_z // 2 : -(diff_z - diff_z // 2), ...]

            if (diff_y - diff_y // 2) != 0:
                output = output[:, :, :, diff_y // 2 : -(diff_y - diff_y // 2), ...]

            if (diff_x - diff_x // 2) != 0:
                output = output[..., diff_x // 2 : -(diff_x - diff_x // 2)]

            # reverse the isotropic sampling
            output_cpu = output.cpu().detach().numpy()  # BCZYX
            output_cpu = zoom(output_cpu, (1, 1, 1 / anisotropy_factor, 1, 1), order=0)
            output = torch.from_numpy(output_cpu).float().to(device)
            if "instance" in sample:
                zoom(
                    instances.cpu().detach().numpy(),
                    (1 / anisotropy_factor, 1, 1),
                    order=0,
                )

            instance_map = cluster.cluster(
                output[0],
                n_sigma=n_sigma,  # 3
                seed_thresh=seed_thresh,
                fg_thresh=fg_thresh,
                min_mask_sum=min_mask_sum,
                min_unclustered_sum=min_unclustered_sum,
                min_object_size=min_object_size,
            )

            if mask_region is not None:
                # ignore predictions in this region prior to saving
                # the tiffs or prior to comparison with GT masks
                instance_map[
                    int(mask_region[0][0]) : int(mask_region[1][0]),
                    int(mask_region[0][1]) : int(mask_region[1][1]),
                    int(mask_region[0][2]) : int(mask_region[1][2]),
                ] = 0  # Z Y X
            else:
                pass

            if "instance" in sample:
                all_results = matching_dataset(
                    y_true=[sample["instance"].squeeze().cpu().detach().numpy()],
                    y_pred=[instance_map.cpu().detach().numpy()],
                    thresh=ap_val,
                    show_progress=False,
                )
                if verbose:
                    print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, "predictions/")):
                    os.makedirs(os.path.join(save_dir, "predictions/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "predictions/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "seeds/")):
                    os.makedirs(os.path.join(save_dir, "seeds/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "seeds/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "ground-truth/")):
                    os.makedirs(os.path.join(save_dir, "ground-truth/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "ground-truth/")
                        )
                    )
                base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
                image_file_names.append(base)

                prediction_file = os.path.join(save_dir, "predictions/", base + ".tif")
                imsave(
                    prediction_file,
                    instance_map.cpu().detach().numpy().astype(np.uint16),
                )

                seeds_file = os.path.join(save_dir, "seeds/", base + ".tif")
                imsave(
                    seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy()
                )

                if "instance" in sample:
                    gt_file = os.path.join(save_dir, "ground-truth/", base + ".tif")
                    imsave(gt_file, sample["instance"].squeeze().cpu().detach().numpy())

        if save_results and "instance" in sample:
            if not os.path.exists(os.path.join(save_dir, "results/")):
                os.makedirs(os.path.join(save_dir, "results/"))
                print(
                    "Created new directory {}".format(
                        os.path.join(save_dir, "results/")
                    )
                )
            txt_file = os.path.join(
                save_dir,
                "results/combined_AP-"
                + "{:.02f}".format(ap_val)
                + "_tta-"
                + str(tta)
                + ".txt",
            )
            with open(txt_file, "w") as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, \
                            min_object_size, seed_thresh, \
                            intersection_threshold, accuracy \n"
                )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_tif = im_name + ".tif"
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(
                            im_name_tif,
                            min_mask_sum,
                            min_unclustered_sum,
                            min_object_size,
                            seed_thresh,
                            ap_val,
                            score,
                        )
                    )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines(
                    "Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(
                        ap_val, np.mean(result_list)
                    )
                )

        if len(result_list) != 0:
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold \
                        = {} at seediness threshold = {:.05f}, is equal to \
                        {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)
                )
            )
            return -np.mean(result_list)
        else:
            return 0.0


def perform_ilp(instance_map_z, min_depth=3, mean_object_size=3363):
    """Perform integer linear programming to minimize a cost function

    Parameters
    ----------
    instance_map_z : list
                    List of instance segmentations (numpy arrays) for all z slices
    min_depth: int, optional
        Allows building edges to slices
        which are `min_depth` slices away.
        This is useful in case for one of the slices, an object != segmented.
    mean_object_size: int, optional
        The mean object size as calculated from the GT masks available
        for the training data.
        This is useful to compute the cost of appearance and disappearance.
        For example, a large segmented object should have a high cost
        appearance and disappearance cost.
    Returns
    -------
    numpy array (D x H x W)
        The instance segmentation for the entire 3D image
    """

    import gurobipy as gp
    import networkx as nx
    from gurobipy import GRB

    G = nx.Graph()
    model = gp.Model()

    X_nodes_appearance = {}
    X_nodes_disappearance = {}
    X_edges = {}

    def calculate_iou(pred, label):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            return 0
        else:
            iou = intersection / union
            return iou

    instance_map = np.asarray(instance_map_z)  # ZYX
    print("Building Graph ...", flush=True)
    # first just add nodes
    for z1 in tqdm(range(instance_map.shape[0]), position=0, leave=True):
        ids1 = np.unique(instance_map[z1])
        ids1 = ids1[ids1 != 0]
        # X_nodes[z1] = {}
        X_nodes_appearance[z1] = {}
        X_nodes_disappearance[z1] = {}
        for id in ids1:
            y, x = np.where(instance_map[z1] == id)
            G.add_node(
                str(z1) + "_" + str(id),
                cost_node=1.0,
                cost_appearance=len(y) / mean_object_size,
                cost_disappearance=len(y) / mean_object_size,
            )  # we just set it to a constant value for now
            if z1 == 0:
                X_nodes_appearance[z1][id] = 1.0
            else:
                X_nodes_appearance[z1][id] = model.addVar(
                    vtype=GRB.BINARY, name="Xapp_%d_%d" % (z1, id)
                )

            if z1 == instance_map.shape[0] - 1:
                X_nodes_disappearance[z1][id] = 1.0
            else:
                X_nodes_disappearance[z1][id] = model.addVar(
                    vtype=GRB.BINARY, name="Xdisapp_%d_%d" % (z1, id)
                )

    print("Nodes added ...", flush=True)

    # next add edges
    for z1 in tqdm(range(instance_map.shape[0]), position=0, leave=True):
        ids1 = np.unique(instance_map[z1])
        ids1 = ids1[ids1 != 0]
        X_edges[z1] = {}
        for z2 in range(z1 + 1, np.minimum(z1 + min_depth + 1, instance_map.shape[0])):
            X_edges[z1][z2] = {}
            ids2 = np.unique(instance_map[z2])
            ids2 = ids2[ids2 != 0]
            for id1 in ids1:
                X_edges[z1][z2][id1] = {}
                for id2 in ids2:
                    iou = calculate_iou(
                        instance_map[z1] == id1, instance_map[z2] == id2
                    )
                    if iou > 0:
                        G.add_edge(
                            str(z1) + "_" + str(id1),
                            str(z2) + "_" + str(id2),
                            cost_edge=1 - iou / (z2 - z1),
                        )
                        X_edges[z1][z2][id1][id2] = model.addVar(
                            vtype=GRB.BINARY, name="X_%d_%d_%d_%d" % (z1, id1, z2, id2)
                        )
    print("Edges added ...", flush=True)

    model.update()
    model.modelSense = GRB.MINIMIZE

    constraints = []
    print("Model initialized ...", flush=True)
    # add CONTINUATION constraint

    # x_node<= x_app + \sum of incoming edges
    # x_node<= x_disapp + \sum of outgoing edges

    for node, cost_dic in G.nodes(data=True):
        # u = z_id
        z1, id1 = node.split("_")
        z1, id1 = int(z1), int(id1)
        incoming_indicators = 0
        for edge in G.edges(node):
            # edge is paired tuple where first entry is node!
            z2, id2 = edge[1].split("_")
            z2, id2 = int(z2), int(id2)
            if z1 < z2:
                pass
            else:
                incoming_indicators += X_edges[z2][z1][id2][id1]
        constraints.append(
            model.addConstr(
                1.0 <= X_nodes_appearance[z1][id1] + incoming_indicators,
                "constraint_continuation_incoming_%d_%d" % (z1, id1),
            )
        )  # TODO

        outgoing_indicators = 0
        for edge in G.edges(node):
            # edge is paired tuple where first entry is node!
            z2, id2 = edge[1].split("_")
            z2, id2 = int(z2), int(id2)
            if z1 < z2:
                outgoing_indicators += X_edges[z1][z2][id1][id2]
            else:
                pass
        constraints.append(
            model.addConstr(
                1.0 <= X_nodes_disappearance[z1][id1] + outgoing_indicators,
                "constraint_continuation_outgoing_%d_%d" % (z1, id1),
            )
        )  # TODO
    print("Constraints added ...", flush=True)

    # finally consider costs
    objective = 0
    for u, cost_dic in G.nodes(data=True):
        # u = z_id
        z1, id1 = u.split("_")
        z1, id1 = int(z1), int(id1)
        objective += (
            cost_dic["cost_appearance"] * X_nodes_appearance[z1][id1]
            + cost_dic["cost_disappearance"] * X_nodes_disappearance[z1][id1]
        )  # TODO

    for u, v, cost_dic in G.edges(data=True):
        # u = z_id
        z1, id1 = u.split("_")
        z2, id2 = v.split("_")
        z1, id1, z2, id2 = int(z1), int(id1), int(z2), int(id2)
        objective += cost_dic["cost_edge"] * X_edges[z1][z2][id1][id2]

    model.setObjective(objective)
    model.update()
    model.optimize()
    print("Model optimized over variables ...", flush=True)

    # reassign ids to the new prediction mask
    instance_map_reassign = np.zeros_like(instance_map)
    count = 1
    new_ids_dic = {}
    for v in tqdm(model.getVars(), position=0, leave=True):
        if v.VarName.split("_")[0] == "X":
            _, z1, id1, z2, id2 = v.VarName.split("_")
            indicator = v.X
            z1, id1, z2, id2 = int(z1), int(id1), int(z2), int(id2)
            if indicator:
                if str(z1) + "_" + str(id1) in new_ids_dic.keys():
                    instance_map_reassign[z1][instance_map[z1] == id1] = new_ids_dic[
                        str(z1) + "_" + str(id1)
                    ]
                    instance_map_reassign[z2][instance_map[z2] == id2] = new_ids_dic[
                        str(z1) + "_" + str(id1)
                    ]
                    new_ids_dic[str(z1) + "_" + str(id1)] = new_ids_dic[
                        str(z1) + "_" + str(id1)
                    ]
                    new_ids_dic[str(z2) + "_" + str(id2)] = new_ids_dic[
                        str(z1) + "_" + str(id1)
                    ]
                else:
                    # start a new id
                    instance_map_reassign[z1][instance_map[z1] == id1] = count
                    instance_map_reassign[z2][instance_map[z2] == id2] = count
                    new_ids_dic[str(z1) + "_" + str(id1)] = count
                    new_ids_dic[str(z2) + "_" + str(id2)] = count
                    count += 1
    print("Ids reassigned  ...", flush=True)
    return instance_map_reassign.astype(np.uint16)


def test_3d_ilp(fg_thresh, *args):
    """Infer the trained 2D model on 3D images and then stitch the
    independent predictions on 2D z-slices by solving a global optimization task

    Parameters
    ----------
    fg_thresh : float
        foreground threshold decides which pixels are considered for
        clustering, based on the predicted seediness scores at these pixels.
    args: dictionary
        Contains other paremeters such as `ap_val`, `seed_thresh` etc
    Returns
    -------
    float
        Average `AP_dsb` over all test images
    """
    (
        seed_thresh,
        ap_val,
        min_mask_sum,
        min_unclustered_sum,
        min_object_size,
        mean_object_size,
        tta,
        model,
        dataset_it,
        save_images,
        save_results,
        save_dir,
        verbose,
        grid_x,
        grid_y,
        grid_z,
        pixel_x,
        pixel_y,
        pixel_z,
        one_hot,
        mask_region,
        n_sigma,
        anisotropy_factor,
        device,
    ) = args

    model.eval()
    # cluster module
    cluster = Cluster(grid_y, grid_x, pixel_y, pixel_x, device)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample["image"]  # B 1 Z Y X
            instance_map_z = []
            seed_z = []
            for z in tqdm(range(im.shape[2]), position=0, leave=True):
                multiple_y = im.shape[3] // 8
                multiple_x = im.shape[4] // 8

                if im.shape[3] % 8 != 0:
                    diff_y = 8 * (multiple_y + 1) - im.shape[3]
                else:
                    diff_y = 0
                if im.shape[4] % 8 != 0:
                    diff_x = 8 * (multiple_x + 1) - im.shape[4]
                else:
                    diff_x = 0
                p2d = (
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                )  # last dim, second last dim

                im_z = F.pad(im[:, :, z], p2d, "reflect")  # B C Y X
                if tta:
                    output = apply_tta_2d(im_z, model)
                else:
                    output = model(im_z)

                instance_map, predictions = cluster.cluster(
                    output[0],
                    n_sigma=n_sigma,
                    fg_thresh=fg_thresh,
                    seed_thresh=seed_thresh,
                    min_mask_sum=min_mask_sum,
                    min_unclustered_sum=min_unclustered_sum,
                    min_object_size=min_object_size,
                )

                # unpad instance_map, instances and images
                if (diff_y - diff_y // 2) != 0:
                    instance_map = instance_map[
                        diff_y // 2 : -(diff_y - diff_y // 2), ...
                    ]
                if (diff_x - diff_x // 2) != 0:
                    instance_map = instance_map[
                        ..., diff_x // 2 : -(diff_x - diff_x // 2)
                    ]
                instance_map_z.append(instance_map.cpu().detach().numpy())
                seed_z.append(torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())
            # correct predictions with gurobi
            instance_map = perform_ilp(
                instance_map_z, mean_object_size=mean_object_size
            )

            if "instance" in sample:
                instances = sample[
                    "instance"
                ].squeeze()  # Z Y X  (squeeze takes away first two dimensions) or DYX

            base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
            image_file_names.append(base)

            if "instance" in sample:
                all_results = matching_dataset(
                    y_true=[instances.cpu().detach().numpy()],
                    y_pred=[instance_map],
                    thresh=ap_val,
                    show_progress=False,
                )
                if verbose:
                    print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, "predictions/")):
                    os.makedirs(os.path.join(save_dir, "predictions/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "predictions/")
                        )
                    )
                if not os.path.exists(os.path.join(save_dir, "seeds/")):
                    os.makedirs(os.path.join(save_dir, "seeds/"))
                    print(
                        "Created new directory {}".format(
                            os.path.join(save_dir, "seeds/")
                        )
                    )

                # save predictions
                base, _ = os.path.splitext(os.path.basename(sample["im_name"][0]))
                instances_file = os.path.join(save_dir, "predictions/", base + ".tif")
                imsave(instances_file, instance_map)

                seeds_file = os.path.join(save_dir, "seeds/", base + ".tif")
                imsave(seeds_file, np.asarray(seed_z).astype(float))

        if save_results and "instance" in sample:
            if not os.path.exists(os.path.join(save_dir, "results/")):
                os.makedirs(os.path.join(save_dir, "results/"))
                print(
                    "Created new directory {}".format(
                        os.path.join(save_dir, "results/")
                    )
                )
            txt_file = os.path.join(
                save_dir,
                "results/combined_AP-"
                + "{:.02f}".format(ap_val)
                + "_tta-"
                + str(tta)
                + ".txt",
            )
            with open(txt_file, "w") as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, \
                            min_object_size, seed_thresh, \
                            intersection_threshold, accuracy \n"
                )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_tif = im_name + ".tif"
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} \
                                {:.02f} {:.05f} \n".format(
                            im_name_tif,
                            min_mask_sum,
                            min_unclustered_sum,
                            min_object_size,
                            seed_thresh,
                            ap_val,
                            score,
                        )
                    )
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines(
                    "Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(
                        ap_val, np.mean(result_list)
                    )
                )

        if len(result_list) != 0:
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold \
                        = {} at seediness threshold \
                        = {:.05f}, is equal to {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)
                )
            )
            return -np.mean(result_list)
        else:
            return 0.0
