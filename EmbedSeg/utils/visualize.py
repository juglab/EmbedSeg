import matplotlib.pyplot as plt
from EmbedSeg.utils.glasbey import Glasbey
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes

def create_color_map(n_colors= 10):
    gb = Glasbey(base_palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], 
             lightness_range=(10,100), 
             hue_range=(10,100), 
             chroma_range=(10,100), 
             no_black=True)
    p = gb.generate_palette(size=n_colors)
    p[0, :] =[0, 0, 0] # make label 0 always black!
    p_ = np.hstack((p, np.ones((p.shape[0], 1))))
    p_ = np.where(p_>0, p_, 0)
    p_ = np.where(p_<=1, p_, 1)
    np.save('../../../cmaps/cmap_'+str(n_colors), p_)
    newcmp = ListedColormap(p_)
    return newcmp

def visualize(image, prediction, ground_truth, embedding, new_cmp):
    plt.figure(figsize=(15,15))
    img_show = image if image.ndim==2 else image[...,0]
    plt.subplot(221); 
    plt.imshow(img_show, cmap='magma'); 
    plt.xlabel('Image')
    plt.axis('off')
    plt.subplot(222); 
    plt.axis('off')
    plt.imshow(ground_truth, cmap=new_cmp, interpolation = 'None')
    plt.xlabel('Ground Truth')
    plt.subplot(223);
    plt.axis('off')
    plt.imshow(embedding,  interpolation = 'None')
    
    plt.subplot(224);  
    plt.axis('off')
    plt.imshow(prediction, cmap=new_cmp, interpolation = 'None')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    
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
    
def visualize_im_pred(image, prediction, new_cmp):
    plt.figure(figsize=(15,15))
    img_show = image if image.ndim==2 else image[...,0]
    plt.subplot(121); 
    plt.imshow(img_show, cmap='magma'); 
    plt.xlabel('Image')
    plt.axis('off')
    plt.subplot(122); 
    plt.axis('off')
    plt.imshow(fill_label_holes(prediction), cmap=new_cmp, interpolation = 'None')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()