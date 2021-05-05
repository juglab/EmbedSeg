import torch
import os
import EmbedSeg.utils.transforms as my_transforms


def create_dataset_dict(data_dir,
                        project_name,
                        size,
                        center,
                        type,
                        normalization_factor,
                        one_hot= False,
                        name='2d',
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
        normalization_factor: int/float
            For 8-bit image, normalization_factor = 255. For 16-bit image, normalization_factor = 65535.
        one_hot: boolean
            If 'True', instance images are perceived as DYX (here each object is encoded as 1 in its individual slice)
            If 'False', instance image is perceived as YX and has the same dimensionality as raw image
        name: string
            One of '2d' or '3d'
        batch_size: int
            Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
        virtual_batch_multiplier: int
            Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
        workers: int
            Number of data-loader workers
    """
    if name =='2d':
        set_transforms =my_transforms.get_transform([
            {
                'name': 'RandomRotationsAndFlips',
                'opts': {
                    'keys': ('image', 'instance', 'label', 'center_image'),
                    'degrees': 90,
                    'one_hot': one_hot,
                }
            },
            {
                'name': 'ToTensorFromNumpy',
                'opts': {
                    'keys': ('image', 'instance', 'label', 'center_image'),
                    'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor, torch.BoolTensor),
                    'normalization_factor': normalization_factor
                }
            },
        ])
    elif name =='3d':
        set_transforms = my_transforms.get_transform([
            {
                'name': 'RandomRotationsAndFlips_3d',
                'opts': {
                    'keys': ('image', 'instance', 'label', 'center_image'),
                    'degrees': 90,
                    'one_hot': one_hot,
                }
            },
            {
                'name': 'ToTensorFromNumpy',
                'opts': {
                    'keys': ('image', 'instance', 'label', 'center_image'),
                    'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor, torch.BoolTensor),
                    'normalization_factor': normalization_factor
                }
            },
        ])
    dataset_dict = {
        'name': name,
        'kwargs': {
            'center': 'center-' + center,
            'data_dir': os.path.join(data_dir,project_name),
            'type': type,
            'size': size,
            'transform': set_transforms,
            'one_hot': one_hot,
        },
        'batch_size': batch_size,
        'virtual_batch_multiplier': virtual_batch_multiplier,
        'workers': workers,

    }
    print("`{}_dataset_dict` dictionary successfully created with: \n -- {} images accessed from {}, "
          "\n -- number of images per epoch equal to {}, "
          "\n -- batch size set at {}, "
          "\n -- virtual batch multiplier set as {}, "
          "\n -- normalization_factor set as {}, "
          "\n -- one_hot set as {}, "
          .format(type, type, os.path.join(data_dir, project_name, type, 'images'), size, batch_size, virtual_batch_multiplier,
                  normalization_factor, one_hot))
    return dataset_dict


def create_test_configs_dict(data_dir,
                             checkpoint_path,
                             save_dir,
                             normalization_factor,
                             tta = True,
                             one_hot= False,
                             ap_val = 0.5,
                             seed_thresh=0.9,
                             min_object_size=36,
                             save_images=True,
                             save_results=True,
                             min_mask_sum=128,
                             min_unclustered_sum=128,
                             cuda=True,
                             n_z = None,
                             n_y = 1024,
                             n_x = 1024,
                             anisotropy_factor = None,
                             l_y = 1,
                             l_x = 1,
                             name = '2d'
                             ):
    """
        Creates `test_configs` dictionary from parameters.
        Parameters
        ----------
        data_dir : str
            Data is read from os.path.join(data_dir, 'test')
        checkpoint_path: str
            This indicates the path to the trained model
        save_dir: str
            This indicates the directory where the results are saved
        normalization_factor: int
            Use 255 for 8-bit images and 65535 for 16-bit images
        tta: boolean
            If True, then use test-time-augmentation
        one_hot: boolean
            If True, then evaluation instance instance images are available in one-hot encoded style (DYX)
        ap_val: float
            Threshold for IOU
        seed_thresh: float
        min_object_size: int
            Ignores objects having pixels less than min_object_size
        save_images: boolean
            If True, then prediction images are saved
        save_results: boolean
            If True, then prediction results are saved in text file
        min_mask_sum: int

        min_unclustered_sum: int

        n_sigma: int
            2 indicates margin along x and margin along y
        num_classes: list
            [4, 1] -> 4 indicates offset in x, offset in y, margin in x, margin in y; 1 indicates seediness score
        cuda: boolean
            True, indicates GPU usage
    """
    if (n_z is None):
        l_z = None
    else:
        l_z = (n_z - 1)/(n_x - 1) * anisotropy_factor

    if name =='2d':
        n_sigma = 2
        num_classes = [4, 1]
        model_name = 'branched_erfnet'
    elif name=='3d':
        n_sigma = 3
        num_classes= [6, 1]
        model_name = 'branched_erfnet_3d'

    test_configs = dict(
        ap_val=ap_val,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
        n_sigma=n_sigma,
        tta=tta,
        seed_thresh=seed_thresh,
        cuda=cuda,
        save_results=save_results,
        save_images=save_images,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
        grid_x = n_x,
        grid_y = n_y,
        grid_z = n_z,
        pixel_x = l_x,
        pixel_y = l_y,
        pixel_z = l_z,
        name = name,
        dataset={
            'name': name,
            'kwargs': {
                'data_dir': data_dir,
                'type': 'test',
                'transform': my_transforms.get_transform([
                    {
                        'name': 'ToTensorFromNumpy',
                        'opts': {
                            'keys': ('image', 'instance', 'label'),
                            'type': (torch.FloatTensor, torch.ShortTensor, torch.ShortTensor),
                            'normalization_factor': normalization_factor
                        }
                    },
                ]),
                'one_hot': one_hot,
            }
        },

        model={
            'name': model_name,
            'kwargs': {
                'num_classes': num_classes,
            }
        }
    )
    print(
        "`test_configs` dictionary successfully created with: "
        "\n -- evaluation images accessed from {}, "
        "\n -- trained weights accessed from {}, "
        "\n -- seediness threshold set at {}, "
        "\n -- output directory chosen as {}".format(
            data_dir, checkpoint_path, seed_thresh, save_dir))
    return test_configs


def create_model_dict(input_channels, num_classes=[4, 1], name='2d'):
    """
        Creates `model_dict` dictionary from parameters.
        Parameters
        ----------
        input_channels: int
            1 indicates gray-channle image, 3 indicates RGB image.
        num_classes: list
            [4, 1] -> 4 indicates offset in x, offset in y, margin in x, margin in y; 1 indicates seediness score
        name: string
    """
    model_dict = {
        'name': 'branched_erfnet' if name=='2d' else 'branched_erfnet_3d',
        'kwargs': {
            'num_classes': num_classes,
            'input_channels': input_channels,
        }
    }
    print(
        "`model_dict` dictionary successfully created with: \n -- num of classes equal to {}, \n -- input channels equal to {}, \n -- name equal to {}".format(
            input_channels, num_classes, name))
    return model_dict


def create_loss_dict(foreground_weight = 10, n_sigma = 2, w_inst = 1, w_var = 10, w_seed = 1):
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
    loss_dict = {'lossOpts': {'n_sigma': n_sigma,
                              'foreground_weight': foreground_weight,
                              },
                 'lossW': {'w_inst': w_inst,
                           'w_var': w_var,
                           'w_seed': w_seed,
                           },
                 }
    print(
        "`loss_dict` dictionary successfully created with: \n -- foreground weight equal to {:.3f}, \n -- w_inst equal to {}, \n -- w_var equal to {}, \n -- w_seed equal to {}".format(
            foreground_weight, w_inst, w_var, w_seed))
    return loss_dict


def create_configs(save_dir,
                   resume_path,
                   one_hot = False,
                   display=False,
                   display_embedding = False,
                   display_it=5,
                   n_epochs=200,
                   train_lr=5e-4,
                   cuda=True,
                   save=True,
                   n_z = None,
                   n_y = 1024,
                   n_x = 1024,
                   anisotropy_factor = None,
                   l_y = 1,
                   l_x = 1,

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
            If 'True', instance images are perceived as DYX (here each object is encoded as 1 in its individual slice)
            If 'False', instance image is perceived as YX and has the same dimensionality as raw image
        display: boolean
            If 'True', then realtime display of images, ground-truth, predictions are shown
        display_embedding: boolean
            If False, it suppresses embedding image
        display_it: int
            Shows display every n training/val steps (display_it = n)
        n_epochs: int
            Total number of epochs
        train_lr: float
            Starting learning rate
        cuda: boolean
            If True, use GPU
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
    """
    if (n_z is None):
        l_z = None
    else:
        l_z = (n_z - 1)/(n_x - 1) * anisotropy_factor

    configs = dict(train_lr = train_lr,
                   n_epochs = n_epochs,
                   cuda = cuda,
                   display = display,
                   display_embedding = display_embedding,
                   display_it = display_it,
                   save = save,
                   save_dir = save_dir,
                   resume_path = resume_path,
                   grid_z = n_z,
                   grid_y = n_y,
                   grid_x = n_x,
                   pixel_z = l_z,
                   pixel_y = l_y,
                   pixel_x = l_x,
                   one_hot=one_hot)
    print(
        "`configs` dictionary successfully created with: "
        "\n -- n_epochs equal to {}, "
        "\n -- display equal to {}, "
        "\n -- save_dir equal to {}, "
        "\n -- n_z equal to {}, "
        "\n -- n_y equal to {}, "
        "\n -- n_x equal to {}, "
        "\n -- one_hot equal to {}, "
        .format(n_epochs, display, save_dir, n_z, n_y, n_x, one_hot))
    return configs
