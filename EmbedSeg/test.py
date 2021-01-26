import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from EmbedSeg.datasets import get_dataset
from EmbedSeg.models import get_model
from EmbedSeg.utils.utils import Cluster, prepare_embedding_for_test_image
from EmbedSeg.utils.utils2 import matching_dataset, obtain_AP_one_hot

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
    im_numpy = im_numpy[np.newaxis, np.newaxis, ...]
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
            if im.shape[2]%4!=0:
                multiple_y = im.shape[2]//4
                diff_y = im.shape[2] - 4*multiple_y
                m = torch.nn.ZeroPad2d((0, 0, diff_y//2, diff_y//2))
                im = m(im)
                instances = m(instances)
            if im.shape[3]%4!=0:
                multiple_x = im.shape[3] // 4
                diff_x = im.shape[3] - 4 * multiple_x
                m = torch.nn.ZeroPad2d((diff_x // 2, diff_x // 2, 0, 0))
                im = m(im)
                instances = m(instances)


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

            center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic = prepare_embedding_for_test_image(instance_map = instance_map, output = output, grid_x = grid_x, grid_y = grid_y, pixel_x = pixel_x, pixel_y =pixel_y, predictions =predictions)

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            imageFileNames.append(base)

            if (one_hot):
                instances_integer = get_instance_map(instances)
                input_max, b = torch.max(instances_integer, 0)
                sc = obtain_AP_one_hot(gt_image = instances.cpu().detach().numpy(), prediction_image = instance_map.cpu().detach().numpy(), ap_val=ap_val)
                if (verbose):
                    print("Accuracy: {:.03f}".format(sc), flush=True)
                resultList.append(sc)
            else:
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
                if not os.path.exists(os.path.join(save_dir, 'embedding/')):
                    os.makedirs(os.path.join(save_dir, 'embedding/'))
                    print("Created new directory {}".format(os.path.join(save_dir, 'embedding/')))


                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                instances_file = os.path.join(save_dir, 'predictions/', base + '.tif')
                imsave(instances_file, instance_map.cpu().detach().numpy())
                gt_file = os.path.join(save_dir, 'ground-truth/', base + '.tif')
                imsave(gt_file, instances.cpu().detach().numpy())
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
