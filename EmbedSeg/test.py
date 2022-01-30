import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import Cluster, Cluster_3d
from EmbedSeg.utils.utils2 import matching_dataset, obtain_AP_one_hot

torch.backends.cudnn.benchmark = True
import numpy as np
from tifffile import imsave
from EmbedSeg.utils.test_time_augmentation import apply_tta_2d, apply_tta_3d
from scipy.ndimage import zoom
from scipy.optimize import minimize_scalar, linear_sum_assignment
from skimage.segmentation import relabel_sequential


def begin_evaluating(test_configs, optimize=False, maxiter=10, verbose=False, mask_region=None):
    """
    :param test_configs: dictionary containing keys such as `n_sigma`, `ap_val` etc
    :param verbose: if verbose=True, then average precision for each image is shown
    :param mask_region: list of lists. Specify as [[mask_start_z, mask_start_y, mask_start_x], [mask_end_z, mask_end_y, mask_end_x]].
    where, mask_start_x etc are pixel coordinates of the cuboidal region which should be masked
    :param mask_intensity: insert this value in the masked region of the image
    :param avg_bg: Average background image intensity in the train and val images
    :return:
    """
    n_sigma = test_configs['n_sigma']
    ap_val = test_configs['ap_val']
    min_mask_sum = test_configs['min_mask_sum']
    min_unclustered_sum = test_configs['min_unclustered_sum']
    min_object_size = test_configs['min_object_size']
    tta = test_configs['tta']
    seed_thresh = test_configs['seed_thresh']
    fg_thresh = test_configs['fg_thresh']
    save_images = test_configs['save_images']
    save_results = test_configs['save_results']
    save_dir = test_configs['save_dir']
    anisotropy_factor = test_configs['anisotropy_factor']
    grid_x = test_configs['grid_x']
    grid_y = test_configs['grid_y']
    grid_z = test_configs['grid_z']
    pixel_x = test_configs['pixel_x']
    pixel_y = test_configs['pixel_y']
    pixel_z = test_configs['pixel_z']
    one_hot = test_configs['dataset']['kwargs']['one_hot']

    # set device
    device = torch.device("cuda:0" if test_configs['cuda'] else "cpu")

    # dataloader
    dataset = get_dataset(test_configs['dataset']['name'], test_configs['dataset']['kwargs'])
    dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4,
                                             pin_memory=True if test_configs['cuda'] else False)

    # load model
    model = get_model(test_configs['model']['name'], test_configs['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)

    # load snapshot
    if os.path.exists(test_configs['checkpoint_path']):
        state = torch.load(test_configs['checkpoint_path'])
        model.load_state_dict(state['model_state_dict'], strict=True)
    else:
        assert False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path'])

    # test on evaluation images:
    result_dic = {}
    if (test_configs['name'] == '2d'):
        args = (fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta,
                model, dataset_it, save_images, save_results, save_dir, verbose, grid_x,
                grid_y, pixel_x, pixel_y, one_hot, n_sigma)
        if optimize:
            result = minimize_scalar(fun=test, bounds=(fg_thresh, 1.0), method='bounded',
                                     args=args, options={'maxiter': maxiter})
            result_dic['seed_thresh'] = result.x
            result_dic['fg_thresh'] = fg_thresh
            result_dic['AP_dsb_05'] = -result.fun
            print("Optimal seed thresh parameter calculated equal to {:.05f}".format(result.x))
            print("AP_dsb at IoU threshold = {} with seed_thresh = {:.05f} equals {:.05f}".format(0.50, result.x,
                                                                                                  -result.fun))
        else:
            result = test(seed_thresh, *args)
            result_dic['seed_thresh'] = seed_thresh
            result_dic['AP_dsb_05'] = -result
    elif (test_configs['name'] == '3d'):
        args = (
            fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it, save_images,
            save_results, save_dir,
            verbose, grid_x, grid_y, grid_z, pixel_x, pixel_y, pixel_z, one_hot, mask_region, n_sigma)

        if optimize:
            result = minimize_scalar(fun=test_3d, bounds=(fg_thresh, 1.0), method='bounded', args=args,
                                     options={'maxiter': maxiter})
            result_dic['seed_thresh'] = result.x
            result_dic['fg_thresh'] = fg_thresh
            result_dic['AP_dsb_05'] = -result.fun
            print("Optimal seed thresh parameter calculated equal to {:.05f}".format(result.x))
            print("AP_dsb at IoU threshold = {} with seed_thresh = {:.05f} equals {:.05f}".format(0.50, result.x,
                                                                                                  -result.fun))
        else:
            result = test_3d(seed_thresh, *args)
            result_dic['seed_thresh'] = seed_thresh
            result_dic['AP_dsb_05'] = -result
    elif (test_configs['name'] == '3d_sliced'):
        args = (fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it,
                save_images, save_results, save_dir, verbose, grid_x, grid_y, grid_z,
                pixel_x, pixel_y, pixel_z, one_hot, mask_region, n_sigma, anisotropy_factor)
        if optimize:
            result = minimize_scalar(fun=test_3d_sliced, bounds=(fg_thresh, 1.0), method='bounded', args=args,
                                     options={'maxiter': maxiter})
            result_dic['seed_thresh'] = result.x
            result_dic['fg_thresh'] = fg_thresh
            result_dic['AP_dsb_05'] = -result.fun
        else:
            result = test_3d_sliced(seed_thresh, *args)
            result_dic['seed_thresh'] = seed_thresh
            result_dic['fg_thresh'] = fg_thresh
            result_dic['AP_dsb_05'] = -result
    return result_dic


def stitch_3d(instance_map_tile, instance_map_current, z_tile=None, y_tile=None, x_tile=None, last=1, num_overlap_slices=4):
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

        instance_map_tile = np.pad(instance_map_tile,
                                   ((z_tile, np.maximum(0, instance_map_current.shape[0] - z_tile - D)),
                                    (y_tile, np.maximum(0, instance_map_current.shape[1] - y_tile - H)),
                                    (x_tile, np.maximum(0, instance_map_current.shape[2] - x_tile - W))))

        # ensure that it has the same shape as instance_map_current
        instance_map_tile = instance_map_tile[:instance_map_current.shape[0], :instance_map_current.shape[1],
                            :instance_map_current.shape[2]]
        mask_overlap = np.zeros_like(instance_map_tile)

        if z_tile == 0 and y_tile == 0 and x_tile == 0:
            ids_tile = np.unique(instance_map_tile)
            ids_tile = ids_tile[ids_tile != 0]
            instance_map_current[:instance_map_tile.shape[0], :instance_map_tile.shape[1],
            :instance_map_tile.shape[2]] = instance_map_tile
            last = len(ids_tile) + 1
        else:
            if x_tile != 0 and y_tile == 0 and z_tile == 0:
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + H, x_tile:x_tile + num_overlap_slices] = 1
            elif x_tile == 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + num_overlap_slices, x_tile:x_tile + W] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile == 0:
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + H, x_tile:x_tile + num_overlap_slices] = 1
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + num_overlap_slices, x_tile:x_tile + W] = 1
            elif x_tile ==0 and y_tile == 0 and z_tile !=0:
                mask_overlap[z_tile:z_tile + num_overlap_slices, y_tile:y_tile + H, x_tile:x_tile + W] = 1
            elif x_tile != 0 and y_tile == 0 and z_tile != 0:
                mask_overlap[z_tile:z_tile + num_overlap_slices, y_tile:y_tile + H, x_tile:x_tile + W] = 1
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + H, x_tile:x_tile + num_overlap_slices] = 1
            elif x_tile == 0 and y_tile !=0 and z_tile !=0:
                mask_overlap[z_tile:z_tile + num_overlap_slices, y_tile:y_tile + H, x_tile:x_tile + W] = 1
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + num_overlap_slices, x_tile:x_tile + W] = 1
            elif x_tile != 0 and y_tile != 0 and z_tile != 0:
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + H, x_tile:x_tile + num_overlap_slices] = 1
                mask_overlap[z_tile:z_tile + D, y_tile:y_tile + num_overlap_slices, x_tile:x_tile + W] = 1
                mask_overlap[z_tile:z_tile + num_overlap_slices, y_tile:y_tile + H, x_tile:x_tile + W] = 1

            # identify ids in the complete tile, not just the overlap region,
            ids_tile_all = np.unique(instance_map_tile)
            ids_tile_all = ids_tile_all[ids_tile_all != 0]

            # identify ids in the the overlap region,
            ids_tile_overlap = np.unique(instance_map_tile * mask_overlap)
            ids_tile_overlap = ids_tile_overlap[ids_tile_overlap != 0]

            # identify ids not in overlap region
            ids_tile_notin_overlap = np.setdiff1d(ids_tile_all, ids_tile_overlap)

            # identify ids in `instance_map_current` but only in the overlap region
            instance_map_current_masked = torch.from_numpy(instance_map_current * mask_overlap).cuda()

            ids_current = torch.unique(instance_map_current_masked).cpu().detach().numpy()
            ids_current = ids_current[ids_current != 0]

            IoU_table = np.zeros((len(ids_tile_overlap), len(ids_current)))
            instance_map_tile_masked = torch.from_numpy(instance_map_tile * mask_overlap).cuda()

            # rows are ids in tile, cols are ids in GT instance map

            for i, id_tile in enumerate(ids_tile_overlap):
                for j, id_current in enumerate(ids_current):

                    intersection = ((instance_map_tile_masked == id_tile)
                                    & (instance_map_current_masked == id_current)).sum()
                    union = ((instance_map_tile_masked == id_tile)
                             | (instance_map_current_masked == id_current)).sum()
                    if union != 0:
                        IoU_table[i, j] = intersection / union
                    else:
                        IoU_table[i, j] = 0.0

            row_indices, col_indices = linear_sum_assignment(-IoU_table)
            matched_indices = np.array(list(zip(row_indices, col_indices)))  # list of (row, col) tuples
            unmatched_indices_tile = np.setdiff1d(np.arange(len(ids_tile_overlap)), row_indices)

            for m in matched_indices:
                if (IoU_table[m[0], m[1]] >= 0.5):  # (tile, current)
                    # whereever the tile is m[0], it should be assigned m[1] in the larger prediction image
                    instance_map_current[instance_map_tile == ids_tile_overlap[m[0]]] = ids_current[m[1]]
                else:
                    # otherwise just take a union of the both ...
                    instance_map_current[instance_map_tile == ids_tile_overlap[m[0]]] = last
                    instance_map_current[instance_map_current == ids_current[m[1]]] = last
                    last += 1
            for index in unmatched_indices_tile:  # not a tuple
                instance_map_current[instance_map_tile == ids_tile_overlap[index]] = last
                last += 1
            for id in ids_tile_notin_overlap:
                instance_map_current[instance_map_tile == id] = last
                last += 1

        return last, instance_map_current
    else:
        return last, instance_map_current  # if there are no ids in tile, then just return


def test(seed_thresh, *args):
    fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it, save_images, \
    save_results, save_dir, verbose, grid_x, grid_y, pixel_x, pixel_y, one_hot, n_sigma = args

    """
    :param verbose: if True, then average prevision is printed out for each image
    :param grid_y:
    :param grid_x:
    :param pixel_y:
    :param pixel_x:
    :param one_hot: True, if the instance masks are encoded in a one-hot fashion
    :param avg_bg: Average Background Image Intensity
    :return:
    """
    model.eval()

    # cluster module
    cluster = Cluster(grid_y, grid_x, pixel_y, pixel_x)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):

            im = sample['image']  # B 1 Y X

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
            p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)  # last dim, second last dim

            im = F.pad(im, p2d, "reflect")
            if ('instance' in sample):
                instances = sample['instance'].squeeze()  # Y X  (squeeze takes away first two dimensions) or DYX
                instances = F.pad(instances, p2d, "constant", 0)

            if (tta):
                output = apply_tta_2d(im, model)
            else:
                output = model(im)

            instance_map, predictions = cluster.cluster(output[0],
                                                        n_sigma=n_sigma,
                                                        fg_thresh=fg_thresh,
                                                        seed_thresh=seed_thresh,
                                                        min_mask_sum=min_mask_sum,
                                                        min_unclustered_sum=min_unclustered_sum,
                                                        min_object_size=min_object_size)

            # unpad instance_map, instances and images
            if (diff_y - diff_y // 2) is not 0:
                instance_map = instance_map[diff_y // 2:-(diff_y - diff_y // 2), ...]
                instances = instances[diff_y // 2:-(diff_y - diff_y // 2), ...]
                im = im[:, :, diff_y // 2:-(diff_y - diff_y // 2), ...]
            if (diff_x - diff_x // 2) is not 0:
                instance_map = instance_map[..., diff_x // 2:-(diff_x - diff_x // 2)]
                instances = instances[..., diff_x // 2:-(diff_x - diff_x // 2)]
                im = im[..., diff_x // 2:-(diff_x - diff_x // 2)]

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            image_file_names.append(base)

            if (one_hot):
                if ('instance' in sample):
                    all_results = obtain_AP_one_hot(gt_image=instances.cpu().detach().numpy(),
                                                    prediction_image=instance_map.cpu().detach().numpy(), ap_val=ap_val)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(all_results), flush=True)
                    result_list.append(all_results)
            else:
                if ('instance' in sample):
                    all_results = matching_dataset(y_true=[instances.cpu().detach().numpy()],
                                                   y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val,
                                                   show_progress=False)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                    result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:

                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))

                # save predictions
                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy().astype(
                    np.uint16))
                # save ground truth
                if ('instance' in sample):
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))
                seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                imsave(seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())

        if save_results and 'instance' in sample:
            if not os.path.exists(os.path.join(save_dir, 'results/')):
                os.makedirs(os.path.join(save_dir, 'results/'))
                print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
            txt_file = os.path.join(save_dir,
                                    'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
            with open(txt_file, 'w') as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_png = im_name + '.png'
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                       min_unclustered_sum,
                                                                                       min_object_size, seed_thresh,
                                                                                       ap_val, score))
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines("Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(ap_val, np.mean(result_list)))

            print(
                "Mean Average Precision (AP_dsb) at IOU threshold = {} at seediness threshold = {:.05f}, is equal to {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)))
        if len(result_list) != 0:
            return -np.mean(result_list)
        else:
            return 0.0


def test_3d(seed_thresh, *args):
    fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it, \
    save_images, save_results, save_dir, verbose, grid_x, grid_y, grid_z, \
    pixel_x, pixel_y, pixel_z, one_hot, mask_region, n_sigma = args

    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample['image']

            last = 1
            D, H, W = im.shape[2], im.shape[3], im.shape[4]
            instance_map_current = np.zeros((D, H, W), dtype=np.uint16)

            num_overlap_slices = 4
            for z in tqdm(range(0, D, grid_z - num_overlap_slices)):
                for y in range(0, H, grid_y - num_overlap_slices):
                    for x in range(0, W, grid_x - num_overlap_slices):
                        im_ = im[:, :, z:z + grid_z, y:y + grid_y, x:x + grid_x]
                        # pad this tile to be a multiple of 8
                        multiple_z = im_.shape[2] // 8
                        multiple_y = im_.shape[3] // 8
                        multiple_x = im_.shape[4] // 8
                        if im_.shape[2] % 8 != 0:
                            diff_z = 8 * (multiple_z + 1) - im_.shape[2]
                        else:
                            diff_z = 0
                        if im_.shape[3] % 8 != 0:
                            diff_y = 8 * (multiple_y + 1) - im_.shape[3]
                        else:
                            diff_y = 0
                        if im_.shape[4] % 8 != 0:
                            diff_x = 8 * (multiple_x + 1) - im_.shape[4]
                        else:
                            diff_x = 0

                        p3d = (0, diff_x, 0, diff_y, 0, diff_z)  # last dim, second last dim, third last dim

                        im_ = F.pad(im_, p3d, 'reflect')

                        if tta:
                            for iter in range(16):
                                if iter == 0:
                                    output_average = apply_tta_3d(im_, model, iter)
                                else:
                                    output_average = 1 / (iter + 1) * (
                                            output_average * iter + apply_tta_3d(im_, model, iter))  # iter
                            output = torch.from_numpy(output_average).float().cuda()
                        else:
                            output = model(im_)

                        instance_map_tile = cluster.cluster(output[0],
                                                            n_sigma=n_sigma,
                                                            fg_thresh=fg_thresh,
                                                            seed_thresh=seed_thresh,
                                                            min_mask_sum=min_mask_sum,
                                                            min_unclustered_sum=min_unclustered_sum,
                                                            min_object_size=min_object_size,
                                                            )
                        # print("clustering done ...")
                        last, instance_map_current = stitch_3d(instance_map_tile.cpu().detach().numpy(),
                                                               instance_map_current, z, y, x, last=last,
                                                               num_overlap_slices=num_overlap_slices)
                        # print("current last object id  is {}".format(last))
            instance_map = torch.from_numpy(instance_map_current.astype(np.int16)).cuda()
            if ('instance' in sample):
                instances = sample['instance'].squeeze()  # Z, Y, X

            if (mask_region is not None):
                # ignore predictions in this region prior to saving the tiffs or prior to comparison with GT masks
                instance_map[int(mask_region[0][0]):int(mask_region[1][0]),
                int(mask_region[0][1]):int(mask_region[1][1]),
                int(mask_region[0][2]):int(mask_region[1][2])] = 0  # Z Y X
            else:
                pass

            if ('instance' in sample):
                all_results = matching_dataset(y_true=[instances.cpu().detach().numpy()],
                                               y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val,
                                               show_progress=False)
                if (verbose):
                    print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                image_file_names.append(base)

                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                if ('instance' in sample):
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))

                seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                imsave(seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())

        if save_results and 'instance' in sample:
            if not os.path.exists(os.path.join(save_dir, 'results/')):
                os.makedirs(os.path.join(save_dir, 'results/'))
                print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
            txt_file = os.path.join(save_dir,
                                    'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
            with open(txt_file, 'w') as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_png = im_name + '.png'
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                       min_unclustered_sum,
                                                                                       min_object_size, seed_thresh,
                                                                                       ap_val, score))
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines("Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(ap_val, np.mean(result_list)))

            print(
                "Mean Average Precision (AP_dsb) at IOU threshold = {} at seediness threshold = {:.05f}, is equal to {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)))

        if len(result_list) != 0:
            return -np.mean(result_list)
        else:
            return 0.0


def test_3d_sliced(seed_thresh, *args):
    fg_thresh, ap_val, min_mask_sum, min_unclustered_sum, min_object_size, tta, model, dataset_it, save_images, save_results, save_dir, \
    verbose, grid_x, grid_y, grid_z, pixel_x, pixel_y, pixel_z, \
    one_hot, mask_region, n_sigma, anisotropy_factor = args

    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)

    with torch.no_grad():
        result_list = []
        image_file_names = []
        for sample in tqdm(dataset_it):
            im = sample['image']  # isotropically expanded image
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
            p3d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2, diff_z // 2,
                   diff_z - diff_z // 2)  # last dim, second last dim, third last dim!

            im = F.pad(im, p3d, "reflect")
            if ('instance' in sample):
                instances = sample['instance'].squeeze()  # isotropically expanded GT instance map
                instances = F.pad(instances, p3d, "constant", 0)

            print("Processing `YX` slices .....")
            # obtain YX slices
            output_YX_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).cuda()
            for z in tqdm(range(im.shape[2]), position=0, leave=True):
                if (tta):
                    output_YX = apply_tta_2d(im[:, :, z, ...], model)
                else:
                    output_YX = model(im[:, :, z, ...])
                output_YX_3d[:, :, z, ...] = output_YX
            print("Processing `ZX` slices .....")
            # obtain ZX slices
            output_ZX_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).cuda()
            for y in tqdm(range(im.shape[3]), position=0, leave=True):
                if (tta):
                    output_ZX = apply_tta_2d(im[:, :, :, y, ...], model)
                else:
                    output_ZX = model(im[:, :, :, y, ...])
                # offset in x, offset in z, sigma in x, sigma in z, seed
                output_ZX_3d[:, :, :, y, ...] = output_ZX
            print("Processing `ZY` slices .....")
            # obtain ZY slices
            output_ZY_3d = torch.zeros(1, 5, im.shape[2], im.shape[3], im.shape[4]).cuda()
            for x in tqdm(range(im.shape[4]), position=0, leave=True):
                if (tta):
                    output_ZY = apply_tta_2d(im[:, :, :, :, x], model)
                else:
                    output_ZY = model(im[:, :, :, :, x])

                # offset in y, offset in z, sigma in y, sigma in z, seed
                output_ZY_3d[:, :, :, :, x] = output_ZY
            output = torch.from_numpy(np.zeros((1, 7, im.shape[2], im.shape[3], im.shape[4]))).float().cuda()

            # handle seediness
            output[:, 6, ...] = 1 / 3 * (output_YX_3d[:, 4, ...] + output_ZX_3d[:, 4, ...] + output_ZY_3d[:, 4, ...])

            # handle offset in X
            output[:, 0, ...] = 1 / 2 * (output_YX_3d[:, 0, ...] + output_ZX_3d[:, 0, ...])

            # handle offset in Y
            output[:, 1, ...] = 1 / 2 * (output_YX_3d[:, 1, ...] + output_ZY_3d[:, 0, ...])

            # handle offset in Z
            output[:, 2, ...] = 1 / 2 * (output_ZX_3d[:, 1, ...] + output_ZY_3d[:, 1, ...])

            # sigma in X
            output[:, 3, ...] = 1 / 2 * (output_YX_3d[:, 2, ...] + output_ZX_3d[:, 2, ...])

            # sigma in Y
            output[:, 4, ...] = 1 / 2 * (output_YX_3d[:, 3, ...] + output_ZY_3d[:, 2, ...])

            # sigma in Z
            output[:, 5, ...] = 1 / 2 * (output_ZX_3d[:, 3, ...] + output_ZY_3d[:, 3, ...])

            # unpad output, instances and image
            if (diff_z - diff_z // 2) is not 0:
                output = output[:, :, diff_z // 2:-(diff_z - diff_z // 2), ...]
                instances = instances[diff_z // 2:-(diff_z - diff_z // 2), ...]
                im = im[:, :, diff_z // 2:-(diff_z - diff_z // 2), ...]
            if (diff_y - diff_y // 2) is not 0:
                output = output[:, :, :, diff_y // 2:-(diff_y - diff_y // 2), ...]
                instances = instances[:, diff_y // 2:-(diff_y - diff_y // 2), ...]
                im = im[:, :, :, diff_y // 2:-(diff_y - diff_y // 2), ...]
            if (diff_x - diff_x // 2) is not 0:
                output = output[..., diff_x // 2:-(diff_x - diff_x // 2)]
                instances = instances[..., diff_x // 2:-(diff_x - diff_x // 2)]
                im = im[..., diff_x // 2:-(diff_x - diff_x // 2)]

            # reverse the isotropic sampling
            output_cpu = output.cpu().detach().numpy()  # BCZYX
            output_cpu = zoom(output_cpu, (1, 1, 1 / anisotropy_factor, 1, 1), order=0)
            output = torch.from_numpy(output_cpu).float().cuda()
            gt_image = zoom(instances.cpu().detach().numpy(), (1 / anisotropy_factor, 1, 1), order=0)
            image = zoom(im[0, 0].cpu().detach().numpy(), (1 / anisotropy_factor, 1, 1), order=0)

            instance_map, predictions = cluster.cluster(output[0],
                                                        n_sigma=n_sigma,  # 3
                                                        seed_thresh=seed_thresh,
                                                        fg_thresh=fg_thresh,
                                                        min_mask_sum=min_mask_sum,
                                                        min_unclustered_sum=min_unclustered_sum,
                                                        min_object_size=min_object_size,
                                                        )

            if (mask_region is not None):
                # ignore predictions in this region prior to saving the tiffs or prior to comparison with GT masks
                instance_map[int(mask_region[0][0]):int(mask_region[1][0]),
                int(mask_region[0][1]):int(mask_region[1][1]),
                int(mask_region[0][2]):int(mask_region[1][2])] = 0  # Z Y X
            else:
                pass

            if (one_hot):
                if ('instance' in sample):
                    all_results = obtain_AP_one_hot(gt_image=gt_image,
                                                    prediction_image=instance_map.cpu().detach().numpy(), ap_val=ap_val)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(all_results), flush=True)
                    result_list.append(all_results)
            else:
                if ('instance' in sample):
                    all_results = matching_dataset(y_true=[gt_image], y_pred=[instance_map.cpu().detach().numpy()],
                                                   thresh=ap_val,
                                                   show_progress=False)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(all_results.accuracy), flush=True)
                    result_list.append(all_results.accuracy)

            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))
                if not os.path.exists(os.path.join(save_dir, 'images/')):
                    os.makedirs(os.path.join(save_dir, 'images/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'images/')))

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                image_file_names.append(base)

                prediction_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(prediction_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                if ('instance' in sample):
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, gt_image.astype(np.uint16))

                seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                imsave(seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())

                im_file = os.path.join(save_dir, 'images/', base + '.tif')
                imsave(im_file, image)

        if save_results and 'instance' in sample:
            if not os.path.exists(os.path.join(save_dir, 'results/')):
                os.makedirs(os.path.join(save_dir, 'results/'))
                print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
            txt_file = os.path.join(save_dir,
                                    'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
            with open(txt_file, 'w') as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(image_file_names):
                    im_name_png = im_name + '.png'
                    score = result_list[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                       min_unclustered_sum,
                                                                                       min_object_size, seed_thresh,
                                                                                       ap_val, score))
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines("Average Precision (AP_dsb)  {:.02f} {:.05f}\n".format(ap_val, np.mean(result_list)))
            print(
                "Mean Average Precision (AP_dsb) at IOU threshold = {} at seediness threshold = {:.05f}, is equal to {:.05f}".format(
                    ap_val, seed_thresh, np.mean(result_list)))

        if len(result_list) != 0:
            return -np.mean(result_list)
        else:
            return 0.0
