from EmbedSeg.utils.generate_crops import generate_center_image, generate_center_image_3d
import pytest
import numpy as np

def test_generate_center_image_1():
    ma = np.zeros((100, 100))
    ma [30:-30, 30:-30] = 2
    ids = np.unique(ma)
    ids = ids[ids!=0]
    center_image = generate_center_image(ma, center='centroid', ids=ids, one_hot = False )
    y, x = np.where(center_image==True)
    assert y==50 and x==50

def test_generate_center_image_2():
    ma = np.zeros((100, 100))
    ma [30:-30, 30:-30] = 2
    ids = np.unique(ma)
    ids = ids[ids!=0]
    center_image = generate_center_image(ma, center='medoid', ids=ids, one_hot = False )
    y, x = np.where(center_image==True)
    assert np.abs(y-50)<=1 and np.abs(x-50)<=1    


