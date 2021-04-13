import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import Cluster, Cluster_3d, prepare_embedding_for_test_image
from EmbedSeg.utils.utils2 import matching_dataset, obtain_AP_one_hot
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
import numpy as np
from tifffile import imsave

from matplotlib.patches import Ellipse


def begin_evaluating(test_configs, verbose=True, mask_region = None, mask_intensity = None):
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
    if(test_configs['name']=='2d'):
        test(verbose = verbose, grid_x = test_configs['grid_x'], grid_y = test_configs['grid_y'],
             pixel_x = test_configs['pixel_x'], pixel_y = test_configs['pixel_y'],
             one_hot = test_configs['dataset']['kwargs']['one_hot'])
    elif(test_configs['name']=='3d'):
        test_3d(verbose=verbose,
                grid_x=test_configs['grid_x'], grid_y=test_configs['grid_y'], grid_z=test_configs['grid_z'],
                pixel_x=test_configs['pixel_x'], pixel_y=test_configs['pixel_y'],pixel_z=test_configs['pixel_z'],
                one_hot=test_configs['dataset']['kwargs']['one_hot'], mask_region= mask_region, mask_intensity=mask_intensity)

def to_cuda(im_numpy):
    im_numpy = im_numpy[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(im_numpy).float().cuda()


def to_numpy(im_cuda):
    return im_cuda.cpu().detach().numpy()


def process_flips(im_numpy):
    im_numpy_correct = im_numpy
    im_numpy_correct[0, 1, ...] = -1 * im_numpy[
        0, 1, ...]  # because flipping is always along y-axis, so only the y-offset gets affected
    return im_numpy_correct

def to_cuda_3d(im_numpy):
    im_numpy = im_numpy[np.newaxis, ...]
    return torch.from_numpy(im_numpy).float().cuda()


def apply_tta_2d(im):
    im_numpy = im.cpu().detach().numpy()
    im0 = im_numpy[0, 0]  # remove batch and channel dimensions # TODO: this is not correct for multi channel images
    im0 = im0[np.newaxis, ...]  # add z dimension --> helps in rotating!

    im1 = np.rot90(im0, 1, (1, 2))
    im2 = np.rot90(im0, 2, (1, 2))
    im3 = np.rot90(im0, 3, (1, 2))
    im4 = np.flip(im0, 1)
    im5 = np.flip(im1, 1)
    im6 = np.flip(im2, 1)
    im7 = np.flip(im3, 1)

    im0_cuda = to_cuda(im0[0, ...])
    im1_cuda = to_cuda(np.ascontiguousarray(im1[0, ...]))
    im2_cuda = to_cuda(np.ascontiguousarray(im2[0, ...]))
    im3_cuda = to_cuda(np.ascontiguousarray(im3[0, ...]))
    im4_cuda = to_cuda(np.ascontiguousarray(im4[0, ...]))
    im5_cuda = to_cuda(np.ascontiguousarray(im5[0, ...]))
    im6_cuda = to_cuda(np.ascontiguousarray(im6[0, ...]))
    im7_cuda = to_cuda(np.ascontiguousarray(im7[0, ...]))

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


def apply_tta_3d_probabilistic(im, flip, times, dir_flip, dir_rot):
    im_numpy = im.cpu().detach().numpy() # BCZYX
    im_transformed = im_numpy[0, :] # remove batch dimension, now CZYX

    if dir_rot == 0:  # rotate about ZY
        temp = np.ascontiguousarray(np.rot90(im_transformed, 2 * times, (
        1, 2)))  # CZYX
    elif dir_rot == 1:  # rotate about YX
        temp = np.ascontiguousarray(np.rot90(im_transformed, times, (2, 3)))
    elif dir_rot == 2:  # rotate about XZ
        temp = np.ascontiguousarray(np.rot90(im_transformed, 2 * times, (3, 1)))

    if flip == 0: # no flip
        pass
    else: # flip
        if dir_flip == 0:
            temp = np.ascontiguousarray(np.flip(temp, axis = 1))  # Z
        elif dir_flip == 1:
            temp = np.ascontiguousarray(np.flip(temp, axis = 2))  # Y
        elif dir_flip == 2:
            temp = np.ascontiguousarray(np.flip(temp, axis = 3))  # X

    output_transformed = model(to_cuda_3d(temp)) #BCZYX = 1 7 Z Y X
    output_transformed_numpy = to_numpy(output_transformed) # BCZYX

    # detransform output
    if flip ==0:
        temp_detransformed_numpy = output_transformed_numpy
    else:
        if dir_flip == 0:
            temp_detransformed_numpy = np.ascontiguousarray(np.flip(output_transformed_numpy, axis=2))  # Z
        elif dir_flip == 1:
            temp_detransformed_numpy = np.ascontiguousarray(np.flip(output_transformed_numpy, axis=3))  # Y
        elif dir_flip == 2:
            temp_detransformed_numpy = np.ascontiguousarray(np.flip(output_transformed_numpy, axis=4))  # X

    if dir_rot == 0:  # rotate about ZY
        temp_detransformed_numpy = np.ascontiguousarray(np.rot90(temp_detransformed_numpy, 2 * times, (
        3, 2)))  # BCZYX
    elif dir_rot == 1:  # rotate about YX
        temp_detransformed_numpy = np.ascontiguousarray(np.rot90(temp_detransformed_numpy, times, (4, 3)))
    elif dir_rot == 2:  # rotate about XZ
        temp_detransformed_numpy = np.ascontiguousarray(np.rot90(temp_detransformed_numpy, 2 * times, (2, 4)))

    #  have to also process the offsets and covariance sensibly
    # for flipping, just the direction of the offset should reverse
    temp_detransformed_numpy_flipped = temp_detransformed_numpy.copy()
    if flip ==0:
        temp_detransformed_numpy_flipped = temp_detransformed_numpy
    else:
        if dir_flip == 0:
            temp_detransformed_numpy_flipped[:, 2, ...] = - temp_detransformed_numpy[:, 2, ...]
        elif dir_flip == 1:
            temp_detransformed_numpy_flipped[:, 1, ...] = - temp_detransformed_numpy[:, 1, ...]
        elif dir_flip == 2:
            temp_detransformed_numpy_flipped[:, 0, ...] = - temp_detransformed_numpy[:, 0, ...]

    temp_detransformed_numpy_correct = temp_detransformed_numpy_flipped.copy()
    if dir_rot == 0 and times%2==1:  # rotate about ZY

        temp_detransformed_numpy_correct[:, 1, ...] = -temp_detransformed_numpy_flipped[:, 1, ...]
        temp_detransformed_numpy_correct[:, 2, ...] = -temp_detransformed_numpy_flipped[:, 2, ...]


    elif dir_rot == 1:  # rotate about YX
        if times ==0:
            pass
        elif times ==1:
            temp_detransformed_numpy_correct[:, 0, ...] = -temp_detransformed_numpy_flipped[:, 1, ...]
            temp_detransformed_numpy_correct[:, 1, ...] = temp_detransformed_numpy_flipped[:, 0, ...]

            temp_detransformed_numpy_correct[:, 3, ...] = temp_detransformed_numpy_flipped[:, 4, ...]
            temp_detransformed_numpy_correct[:, 4, ...] = temp_detransformed_numpy_flipped[:, 3, ...]
        elif times ==2:
            temp_detransformed_numpy_correct[:, 0, ...] = -temp_detransformed_numpy_flipped[:, 0, ...]
            temp_detransformed_numpy_correct[:, 1, ...] = -temp_detransformed_numpy_flipped[:, 1, ...]
        elif times ==3:
            temp_detransformed_numpy_correct[:, 0, ...] = temp_detransformed_numpy_flipped[:, 1, ...]
            temp_detransformed_numpy_correct[:, 1, ...] = -temp_detransformed_numpy_flipped[:, 0, ...]

            temp_detransformed_numpy_correct[:, 3, ...] = temp_detransformed_numpy_flipped[:, 4, ...]
            temp_detransformed_numpy_correct[:, 4, ...] = temp_detransformed_numpy_flipped[:, 3, ...]

    elif dir_rot == 2 and times%2==1:  # rotate about XZ
        temp_detransformed_numpy_correct[:, 2, ...] = -temp_detransformed_numpy_flipped[:, 2, ...]
        temp_detransformed_numpy_correct[:, 0, ...] = -temp_detransformed_numpy_flipped[:, 0, ...]
    return temp_detransformed_numpy_correct



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

            im = F.pad(im, p2d, "constant", 0)
            if('instance' in sample):
                instances = sample['instance'].squeeze()  # Y X  (squeeze takes away first two dimensions) or DYX
                instances = F.pad(instances, p2d, "constant", 0)

            if (tta):
                output = apply_tta_2d(im)
            else:
                output = model(im)


            instance_map, predictions = cluster.cluster(output[0],
                                                        n_sigma=n_sigma,
                                                        seed_thresh=seed_thresh,
                                                        min_mask_sum=min_mask_sum,
                                                        min_unclustered_sum=min_unclustered_sum,
                                                        min_object_size=min_object_size)


            center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, sample_spatial_embedding_y, sigma_x, sigma_y, \
            color_sample_dic, color_embedding_dic = prepare_embedding_for_test_image(instance_map = instance_map, output = output, grid_x = grid_x, grid_y = grid_y,
                                                                                     pixel_x = pixel_x, pixel_y =pixel_y, predictions =predictions)

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            imageFileNames.append(base)

            if (one_hot):
                if ('instance' in sample):
                    results = obtain_AP_one_hot(gt_image = instances.cpu().detach().numpy(), prediction_image = instance_map.cpu().detach().numpy(), ap_val=ap_val)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(results), flush=True)
                    resultList.append(results)
            else:
                if ('instance' in sample):
                    results=matching_dataset(y_true=[instances.cpu().detach().numpy()], y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val, show_progress = False)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(results.accuracy), flush=True)
                    resultList.append(results.accuracy)



            if save_images and ap_val == 0.5:
                if not os.path.exists(os.path.join(save_dir, 'predictions/')):
                    os.makedirs(os.path.join(save_dir, 'predictions/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'predictions/')))
                if not os.path.exists(os.path.join(save_dir, 'ground-truth/')):
                    os.makedirs(os.path.join(save_dir, 'ground-truth/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'ground-truth/')))
                if not os.path.exists(os.path.join(save_dir, 'embedding/')):
                    os.makedirs(os.path.join(save_dir, 'embedding/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'embedding/')))


                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                if ('instance' in sample):
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))
                embedding_file = os.path.join(save_dir, 'embedding/', base + '.tif')
                import matplotlib
                matplotlib.use('Agg')
                fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi =150)
                ax.imshow(instance_map > 0, cmap='gray')

                for i in range(len(color_sample_dic.items())):
                    ax.plot(center_x[i + 1], center_y[i + 1], color=color_embedding_dic[i + 1], marker='x')
                    ax.scatter(samples_x[i + 1], samples_y[i + 1], color=color_sample_dic[i + 1], marker='+')
                    ax.scatter(sample_spatial_embedding_x[i + 1], sample_spatial_embedding_y[i + 1],
                               color=color_embedding_dic[i + 1], marker='.')
                    ellipse = Ellipse((center_x[i + 1], center_y[i + 1]), width=sigma_x[i + 1], height=sigma_y[i + 1],
                                      angle=0, color=color_embedding_dic[i + 1], alpha=0.5)
                    ax.add_artist(ellipse)
                ax.axis('off')
                plt.tight_layout()
                plt.draw()
                plt.savefig(embedding_file)

        if save_results and 'instance' in sample:
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


def test_3d(verbose, grid_x=1024, grid_y=1024, grid_z= 32, pixel_x=1, pixel_y=1, pixel_z = 1, one_hot = False, mask_region = None, mask_intensity = None):
    model.eval()
    # cluster module
    cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)

    with torch.no_grad():
        resultList = []
        imageFileNames = []
        for sample in tqdm(dataset_it):
            im = sample['image']
            if(mask_region is not None and mask_intensity is not None):
                im[:, :, int(mask_region[0][0]):, : int(mask_region[1][1]), int(mask_region[0][2]):] = mask_intensity # B 1 Z Y X
            else:
                pass


            multiple_z =im.shape[2]//8
            multiple_y =im.shape[3]//8
            multiple_x =im.shape[4]//8

            if im.shape[2]%8!=0:
                diff_z= 8 * (multiple_z + 1) - im.shape[2]
            else:
                diff_z = 0
            if im.shape[3]%8!=0:
                diff_y= 8 * (multiple_y + 1) - im.shape[3]
            else:
                diff_y = 0
            if im.shape[4]%8!=0:
                diff_x= 8 * (multiple_x + 1) - im.shape[4]
            else:
                diff_x = 0
            p3d = (diff_x//2, diff_x - diff_x//2, diff_y//2, diff_y -diff_y//2, diff_z//2, diff_z - diff_z//2) # last dim, second last dim, third last dim!

            im = F.pad(im, p3d, "constant", 0)
            if ('instance' in sample):
                instances = sample['instance'].squeeze()
                instances = F.pad(instances, p3d, "constant", 0)

            times_list =   [0, 1, 2, 3,    0, 1, 2, 3,    0, 1, 2, 3,    1, 1, 1, 1,    0, 1, 2, 3,    1, 1, 1, 1] # 0 --> 0 deg, 1 --> 90 deg, 2 --> 180 deg, 3 --> 270 deg
            flip_list =    [0, 0, 0, 0,    1, 1, 1, 1,    1, 1, 1, 1,    0, 0, 0, 0,    1, 1, 1, 1,    0, 0, 0, 0] # 0 --> no, 1 --> yes
            dir_rot_list = [1, 1, 1, 1,    1, 1, 1, 1,    1, 1, 1, 1,    2, 2, 2, 2,    1, 1, 1, 1,    0, 0, 0, 0] # 0 --> ZY plane, 1 --> YX plane, 2--> XZ plane
            dir_flip_list= [0, 0, 0, 0,    0, 0, 0, 0,    2, 2, 2, 2,    0, 0, 0, 0,    1, 1, 1, 1,    0, 0, 0, 0] # 0 --> Z axis, 1 --> Y axis, 2 --> X axis

            if (tta):
                for iter in range(24):
                    times = times_list[iter]
                    flip = flip_list[iter]  # no or yes
                    dir_rot = dir_rot_list[iter]  # 0 --> ZY, 1 --> YX, 2 --> XZ
                    dir_flip = dir_flip_list[iter]  # 0 --> Z , 1--> Y, 2--> X
                    if iter == 0:
                        output_average = apply_tta_3d_probabilistic(im, flip, times, dir_flip, dir_rot) # BCZYX
                    else:
                        output_average = 1/(iter+1)*(output_average * iter + apply_tta_3d_probabilistic(im, flip, times, dir_flip, dir_rot))

                output = torch.from_numpy(output_average).float().cuda()
            else:
                output = model(im)

            instance_map, predictions = cluster.cluster(output[0],
                                                        n_sigma=n_sigma,
                                                        seed_thresh=seed_thresh,
                                                        min_mask_sum=min_mask_sum,
                                                        min_unclustered_sum=min_unclustered_sum,
                                                        min_object_size=min_object_size,
                                                        )



            if (one_hot):
                if ('instance' in sample):
                    sc = obtain_AP_one_hot(gt_image = instances.cpu().detach().numpy(), prediction_image = instance_map.cpu().detach().numpy(), ap_val=ap_val)
                    if (verbose):
                        print("Accuracy: {:.03f}".format(sc), flush=True)
                    resultList.append(sc)
            else:
                if ('instance' in sample):
                    sc=matching_dataset(y_true= [instances.cpu().detach().numpy()], y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val, show_progress = False) # TODO 1 jan
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
                if not os.path.exists(os.path.join(save_dir, 'seeds/')):
                    os.makedirs(os.path.join(save_dir, 'seeds/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'seeds/')))
                if not os.path.exists(os.path.join(save_dir, 'images/')):
                    os.makedirs(os.path.join(save_dir, 'images/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'images/')))


                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                imageFileNames.append(base)

                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy().astype(np.uint16))
                if ('instance' in sample):
                    gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                    imsave(gt_file, instances.cpu().detach().numpy().astype(np.uint16))

                seeds_file = os.path.join(save_dir, 'seeds/', base + '.tif')
                imsave(seeds_file, torch.sigmoid(output[0, -1, ...]).cpu().detach().numpy())

                im_file = os.path.join(save_dir, 'images/', base + '.tif')
                imsave(im_file, im[0,0].cpu().detach().numpy())


        if save_results and 'instance' in sample:
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
