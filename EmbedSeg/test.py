import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import Cluster, prepare_embedding_for_test_image
from EmbedSeg.utils.utils2 import matching_dataset, obtain_AP_one_hot
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import numpy as np
from tifffile import imsave

from matplotlib.patches import Ellipse


def begin_evaluating(test_configs, verbose=True):
    global n_sigma, ap_val, min_mask_sum, min_unclustered_sum, min_object_size
    global tta, seed_thresh, model, dataset_it, save_images, save_results, save_dir

    n_sigma = test_configs['n_sigma']
    ap_val = test_configs['ap_val']
    min_mask_sum = test_configs['min_mask_sum']
    min_unclustered_sum = test_configs['min_unclustered_sum']
    min_object_size = test_configs['min_object_size']
    tta = test_configs['tta']
    seed_thresh = test_configs['seed_thresh']
    save_images = test_configs['save_images']
    save_results = test_configs['save_results']
    save_dir = test_configs['save_dir']

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
        assert (False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path']))

    # test on evaluation images:
    test(verbose = verbose, grid_x = test_configs['grid_x'], grid_y = test_configs['grid_y'], pixel_x = test_configs['pixel_x'], pixel_y = test_configs['pixel_y'], one_hot = test_configs['dataset']['kwargs']['one_hot'])


def to_cuda(im_numpy):
    im_numpy = im_numpy[np.newaxis,  ...]
    return torch.from_numpy(im_numpy).float().cuda()


def to_numpy(im_cuda):
    return im_cuda.cpu().detach().numpy()


def process_flips(im_numpy):
    im_numpy_correct = im_numpy
    im_numpy_correct[0, 1, ...] = -1 * im_numpy[
        0, 1, ...]  # because flipping is always along y-axis, so only the y-offset gets affected
    return im_numpy_correct


def applyTTAComplete(im):
    im_numpy = im.cpu().detach().numpy()
    im0 = im_numpy[0, :]  # remove batch and channel dimensions # TODO: this is not correct for multi channel images

    im1 = np.rot90(im0, 1, (1, 2))
    im2 = np.rot90(im0, 2, (1, 2))
    im3 = np.rot90(im0, 3, (1, 2))
    im4 = np.flip(im0, 1)
    im5 = np.flip(im1, 1)
    im6 = np.flip(im2, 1)
    im7 = np.flip(im3, 1)


    im0_cuda = to_cuda(im0)
    im1_cuda = to_cuda(np.ascontiguousarray(im1))
    im2_cuda = to_cuda(np.ascontiguousarray(im2))
    im3_cuda = to_cuda(np.ascontiguousarray(im3))
    im4_cuda = to_cuda(np.ascontiguousarray(im4))
    im5_cuda = to_cuda(np.ascontiguousarray(im5))
    im6_cuda = to_cuda(np.ascontiguousarray(im6))
    im7_cuda = to_cuda(np.ascontiguousarray(im7))

    output0 = model(im0_cuda)
    output1 = model(im1_cuda)
    output2 = model(im2_cuda)
    output3 = model(im3_cuda)
    output4 = model(im4_cuda)
    output5 = model(im5_cuda)
    output6 = model(im6_cuda)
    output7 = model(im7_cuda)

    # detransform outputs
    output0_numpy = to_numpy(output0)
    output1_numpy = to_numpy(output1)
    output2_numpy = to_numpy(output2)
    output3_numpy = to_numpy(output3)
    output4_numpy = to_numpy(output4)
    output5_numpy = to_numpy(output5)
    output6_numpy = to_numpy(output6)
    output7_numpy = to_numpy(output7)

    # invert rotations and flipping

    output1_numpy = np.rot90(output1_numpy, 1, (3, 2))
    output2_numpy = np.rot90(output2_numpy, 2, (3, 2))
    output3_numpy = np.rot90(output3_numpy, 3, (3, 2))
    output4_numpy = np.flip(output4_numpy, 2)
    output5_numpy = np.flip(output5_numpy, 2)
    output5_numpy = np.rot90(output5_numpy, 1, (3, 2))
    output6_numpy = np.flip(output6_numpy, 2)
    output6_numpy = np.rot90(output6_numpy, 2, (3, 2))
    output7_numpy = np.flip(output7_numpy, 2)
    output7_numpy = np.rot90(output7_numpy, 3, (3, 2))

    # have to also process the offsets and covariance sensibly
    output0_numpy_correct = output0_numpy

    # note rotations are always [cos(theta) sin(theta); -sin(theta) cos(theta)]
    output1_numpy_correct = np.zeros_like(output1_numpy)
    output1_numpy_correct[0, 0, ...] = -output1_numpy[0, 1, ...]
    output1_numpy_correct[0, 1, ...] = output1_numpy[0, 0, ...]
    output1_numpy_correct[0, 2, ...] = output1_numpy[0, 3, ...]
    output1_numpy_correct[0, 3, ...] = output1_numpy[0, 2, ...]
    output1_numpy_correct[0, 4, ...] = output1_numpy[0, 4, ...]

    output2_numpy_correct = np.zeros_like(output2_numpy)
    output2_numpy_correct[0, 0, ...] = -output2_numpy[0, 0, ...]
    output2_numpy_correct[0, 1, ...] = -output2_numpy[0, 1, ...]
    output2_numpy_correct[0, 2, ...] = output2_numpy[0, 2, ...]
    output2_numpy_correct[0, 3, ...] = output2_numpy[0, 3, ...]
    output2_numpy_correct[0, 4, ...] = output2_numpy[0, 4, ...]

    output3_numpy_correct = np.zeros_like(output3_numpy)
    output3_numpy_correct[0, 0, ...] = output3_numpy[0, 1, ...]
    output3_numpy_correct[0, 1, ...] = -output3_numpy[0, 0, ...]
    output3_numpy_correct[0, 2, ...] = output3_numpy[0, 3, ...]
    output3_numpy_correct[0, 3, ...] = output3_numpy[0, 2, ...]
    output3_numpy_correct[0, 4, ...] = output3_numpy[0, 4, ...]

    output4_numpy_correct = process_flips(output4_numpy)

    output5_numpy_flipped = process_flips(output5_numpy)
    output5_numpy_correct = np.zeros_like(output5_numpy_flipped)
    output5_numpy_correct[0, 0, ...] = -output5_numpy_flipped[0, 1, ...]
    output5_numpy_correct[0, 1, ...] = output5_numpy_flipped[0, 0, ...]
    output5_numpy_correct[0, 2, ...] = output5_numpy_flipped[0, 3, ...]
    output5_numpy_correct[0, 3, ...] = output5_numpy_flipped[0, 2, ...]
    output5_numpy_correct[0, 4, ...] = output5_numpy_flipped[0, 4, ...]

    output6_numpy_flipped = process_flips(output6_numpy)
    output6_numpy_correct = np.zeros_like(output6_numpy_flipped)
    output6_numpy_correct[0, 0, ...] = -output6_numpy_flipped[0, 0, ...]
    output6_numpy_correct[0, 1, ...] = -output6_numpy_flipped[0, 1, ...]
    output6_numpy_correct[0, 2, ...] = output6_numpy_flipped[0, 2, ...]
    output6_numpy_correct[0, 3, ...] = output6_numpy_flipped[0, 3, ...]
    output6_numpy_correct[0, 4, ...] = output6_numpy_flipped[0, 4, ...]

    output7_numpy_flipped = process_flips(output7_numpy)
    output7_numpy_correct = np.zeros_like(output7_numpy_flipped)
    output7_numpy_correct[0, 0, ...] = output7_numpy_flipped[0, 1, ...]
    output7_numpy_correct[0, 1, ...] = -output7_numpy_flipped[0, 0, ...]
    output7_numpy_correct[0, 2, ...] = output7_numpy_flipped[0, 3, ...]
    output7_numpy_correct[0, 3, ...] = output7_numpy_flipped[0, 2, ...]
    output7_numpy_correct[0, 4, ...] = output7_numpy_flipped[0, 4, ...]

    output = np.concatenate((output0_numpy_correct, output1_numpy_correct, output2_numpy_correct, output3_numpy_correct,
                             output4_numpy_correct, output5_numpy_correct, output6_numpy_correct,
                             output7_numpy_correct), 0)
    output = np.mean(output, 0, keepdims=True)  # 1 5 Y X
    return torch.from_numpy(output).float().cuda()


def get_instance_map(image):
    for z in range(image.shape[0]):
        image[z, image[z, ...] == 1] = z + 1
    return image


def test(verbose, grid_y=1024, grid_x=1024, pixel_y=1, pixel_x=1, one_hot = False):
    model.eval()

    # cluster module
    cluster = Cluster(grid_y, grid_x, pixel_y, pixel_x)

    with torch.no_grad():
        resultList = []
        imageFileNames = []
        for sample in tqdm(dataset_it):

            im = sample['image']  # B 1 Y X
            instances = sample['instance'].squeeze()  # Y X  (squeeze takes away first two dimensions) or DYX



            if im.ndimension()==5:
                instance_map = []
                instance_temp = []
                for z in range(im.shape[2]):
                    im_z = im[:, :, z, ...]
                    instances_z = instances[z, ...]
                    multiple_y = im_z.shape[2] // 8
                    multiple_x = im_z.shape[3] // 8

                    if im_z.shape[2] % 8 != 0:
                        diff_y = 8 * (multiple_y + 1) - im_z.shape[2]
                    else:
                        diff_y = 0
                    if im_z.shape[3] % 8 != 0:
                        diff_x = 8 * (multiple_x + 1) - im_z.shape[3]
                    else:
                        diff_x = 0
                    p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)

                    im_z = F.pad(im_z, p2d, "constant", 0)
                    instances_z = F.pad(instances_z, p2d, "constant", 0)
                    if (tta):
                        output = applyTTAComplete(im_z)
                    else:
                        output = model(im_z)

                    instance_map_z, predictions_z = cluster.cluster(output[0],
                                                                n_sigma=n_sigma,
                                                                seed_thresh=seed_thresh,
                                                                min_mask_sum=min_mask_sum,
                                                                min_unclustered_sum=min_unclustered_sum,
                                                                min_object_size=min_object_size)

                    instance_map.append(instance_map_z.unsqueeze(0))
                    instance_temp.append(instances_z.unsqueeze(0))

                    sc = matching_dataset([instance_map_z.cpu().detach().numpy()], [instances_z.cpu().detach().numpy()],
                                          thresh=ap_val, show_progress=False)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(sc.accuracy), flush=True)
                    resultList.append(sc.accuracy)

                instance_temp = torch.cat(instance_temp)
                instance_map=torch.cat(instance_map)
            else:

                multiple_y = im.shape[2] // 8
                multiple_x = im.shape[3] // 8

                if im.shape[2] % 8 != 0:
                    diff_y = 8 * (multiple_y + 1) - im.shape[3]
                else:
                    diff_y = 0
                if im.shape[3] % 8 != 0:
                    diff_x = 8 * (multiple_x + 1) - im.shape[4]
                else:
                    diff_x = 0
                p2d = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
                im = F.pad(im, p2d, "constant", 0)
                instances = F.pad(instances, p2d, "constant", 0)



                if (tta):
                    output = applyTTAComplete(im)
                else:
                    output = model(im)


                instance_map, predictions = cluster.cluster(output[0],
                                                            n_sigma=n_sigma,
                                                            seed_thresh=seed_thresh,
                                                            min_mask_sum=min_mask_sum,
                                                            min_unclustered_sum=min_unclustered_sum,
                                                            min_object_size=min_object_size)
                sc=matching_dataset([instance_map.cpu().detach().numpy()], [instances.cpu().detach().numpy()], thresh=ap_val, show_progress = False)
                if (verbose):
                    print("Accuracy: {:.03f}".format(sc.accuracy), flush=True)
                resultList.append(sc.accuracy)



            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))


                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                instance_map = instance_map[:, diff_y // 2: -(diff_y - diff_y // 2), :]
                imsave(instances_file, instance_map.cpu().detach().numpy())
                gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                imsave(gt_file, instances.cpu().detach().numpy())


        if save_results:
            if not os.path.exists(os.path.join(save_dir, 'results/')):
                os.makedirs(os.path.join(save_dir, 'results/'))
                print("Created new directory {}".format(os.path.join(save_dir, 'results/')))
            txt_file = os.path.join(save_dir, 'results/combined_AP-' + '{:.02f}'.format(ap_val) + '_tta-' + str(tta) + '.txt')
            with open(txt_file, 'w') as f:
                f.writelines(
                    "image_file_name, min_mask_sum, min_unclustered_sum, min_object_size, seed_thresh, intersection_threshold, accuracy \n")
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                for ind, im_name in enumerate(imageFileNames):
                    im_name_png = im_name + '.png'
                    score = resultList[ind]
                    f.writelines(
                        "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.05f} \n".format(im_name_png, min_mask_sum,
                                                                                       min_unclustered_sum,
                                                                                       min_object_size, seed_thresh,
                                                                                       ap_val, score))
                f.writelines("+++++++++++++++++++++++++++++++++\n")
                f.writelines("Average Precision (AP)  {:.02f} {:.05f}\n".format(ap_val, np.mean(resultList)))

        print("Mean Average Precision at IOU threshold = {}, is equal to {:.05f}".format(ap_val, np.mean(resultList)))
