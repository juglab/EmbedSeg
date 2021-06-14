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
from EmbedSeg.utils.test_time_augmentation import apply_tta_2d, apply_tta_3d


def begin_evaluating(test_configs, verbose=True, mask_region = None, mask_intensity = None, avg_bg = None):
    """
    :param test_configs: dictionary containing keys such as `n_sigma`, `ap_val` etc
    :param verbose: if verbose=True, then average precision for each image is shown
    :param mask_region: list of lists. Specify as [[mask_start_z, mask_start_y, mask_start_x], [mask_end_z, mask_end_y, mask_end_x]].
    where, mask_start_x etc are pixel coordinates of the cuboidal region which should be masked
    :param mask_intensity: insert this value in the masked region of the image
    :param avg_bg: Average background image intensity in the train and val images
    :return:
    """
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
             one_hot = test_configs['dataset']['kwargs']['one_hot'], avg_bg = avg_bg, n_sigma=n_sigma)
    elif(test_configs['name']=='3d'):
        test_3d(verbose=verbose,
                grid_x=test_configs['grid_x'], grid_y=test_configs['grid_y'], grid_z=test_configs['grid_z'],
                pixel_x=test_configs['pixel_x'], pixel_y=test_configs['pixel_y'],pixel_z=test_configs['pixel_z'],
                one_hot=test_configs['dataset']['kwargs']['one_hot'], mask_region= mask_region, mask_intensity=mask_intensity, avg_bg = avg_bg)



def test(verbose, grid_y=1024, grid_x=1024, pixel_y=1, pixel_x=1, one_hot = False, avg_bg = 0, n_sigma = 2):
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

            im = F.pad(im, p2d, "constant", avg_bg)
            if('instance' in sample):
                instances = sample['instance'].squeeze()  # Y X  (squeeze takes away first two dimensions) or DYX
                instances = F.pad(instances, p2d, "constant", 0)

            if (tta):
                output = apply_tta_2d(im, model)
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
                                                                                     pixel_x = pixel_x, pixel_y =pixel_y, predictions =predictions, n_sigma = n_sigma)

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


def test_3d(verbose, grid_x=1024, grid_y=1024, grid_z= 32, pixel_x=1, pixel_y=1, pixel_z = 1, one_hot = False, mask_region = None, mask_intensity = None, avg_bg = 0):
    
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

            im = F.pad(im, p3d, "constant", avg_bg)
            if ('instance' in sample):
                instances = sample['instance'].squeeze()
                instances = F.pad(instances, p3d, "constant", 0)
            
            if (tta):
                for iter in range(16):
                    if iter == 0:
                        output_average = apply_tta_3d(im, model, iter)  # iter
                    else:
                        output_average = 1 / (iter + 1) * (output_average * iter + apply_tta_3d(im, model, iter))  # iter
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
                    sc=matching_dataset(y_true= [instances.cpu().detach().numpy()], y_pred=[instance_map.cpu().detach().numpy()], thresh=ap_val, show_progress = False) 
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
