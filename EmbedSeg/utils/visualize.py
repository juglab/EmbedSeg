import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
from glob import glob
from skimage.segmentation import relabel_sequential
from EmbedSeg.utils.glasbey import Glasbey


def create_color_map(n_colors=10):
    gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                 lightness_range=(10, 100),
                 hue_range=(10, 100),
                 chroma_range=(10, 100),
                 no_black=True)
    p = gb.generate_palette(size=n_colors)
    p[0, :] = [0, 0, 0]  # make label 0 always black!
    p_ = np.hstack((p, np.ones((p.shape[0], 1))))
    p_ = np.where(p_ > 0, p_, 0)
    p_ = np.where(p_ <= 1, p_, 1)
    return p_


def visualize(image, prediction, ground_truth, embedding, new_cmp):
    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 16,
            }
    plt.figure(figsize=(15, 15))
    img_show = image if image.ndim == 2 else image[0, ...]
    plt.subplot(221);
    plt.imshow(img_show, cmap='magma');
    plt.text(30, 30, "IM", fontdict=font)
    plt.xlabel('Image')
    plt.axis('off')
    if (ground_truth is not None):
        plt.subplot(222);
        plt.axis('off')
        plt.imshow(ground_truth, cmap=new_cmp, interpolation='None')
        plt.text(30, 30, "GT", fontdict=font)
        plt.xlabel('Ground Truth')
    plt.subplot(223);
    plt.axis('off')
    plt.imshow(embedding, interpolation='None')
    plt.subplot(224);
    plt.axis('off')
    plt.imshow(prediction, cmap=new_cmp, interpolation='None')
    plt.text(30, 30, "PRED", fontdict=font)
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()


def visualize_many_crops(data_dir, project_name, train_val_dir, center, n_images, new_cmp):
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 16,
            }
    im_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir)+'/images/*.tif'))
    ma_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir)+ '/masks/*.tif'))
    center_filenames = sorted(glob(os.path.join(data_dir, project_name, train_val_dir)+ '/center-'+center+'/*.tif'))
    indices = np.random.randint(0, len(im_filenames), n_images)
    fig = plt.figure(constrained_layout=False, figsize=(16, 10))
    spec = gridspec.GridSpec(ncols=n_images, nrows=3, figure=fig)
    for i, index in enumerate(indices):
        ax0 = fig.add_subplot(spec[0, i])
        ax0.imshow(tifffile.imread(im_filenames[index])[0], cmap='magma', interpolation='None')
        ax0.axes.get_xaxis().set_visible(False)
        ax0.set_yticklabels([])
        ax0.set_yticks([])
        if i==0:
            ax0.set_ylabel('IM', fontdict=font)
        ax1 = fig.add_subplot(spec[1, i])
        label, _, _ = relabel_sequential(tifffile.imread(ma_filenames[index]))
        ax1.imshow(label, cmap=new_cmp, interpolation='None')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        if i==0:
            ax1.set_ylabel('MASK', fontdict=font)
        ax2 = fig.add_subplot(spec[2, i])
        ax2.imshow(tifffile.imread(center_filenames[index]), cmap='gray', interpolation='None')
        ax2.axes.get_xaxis().set_visible(False)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        if i==0:
            ax2.set_ylabel('CENTER', fontdict=font)
        plt.tight_layout(pad=0, h_pad=0)
    plt.show()