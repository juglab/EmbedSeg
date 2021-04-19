import os
import urllib.request
import zipfile
import shutil
from glob import glob
import numpy as np
import tifffile
from tqdm import tqdm
import json

def extract_data(zip_url, project_name, data_dir='../../../data/'):
    """
        Extracts data from `zip_url` to the location identified by `data_dir` and `project_name` parameters.
        Parameters
        ----------
        zip_url: string
            Indicates the url
        project_name: string
            Indicates the path to the sub-directory at the location identified by the parameter `data_dir`
        data_dir: string
            Indicates the path to the directory where the data should be saved.

    """
    zip_path = os.path.join(data_dir, project_name + '.zip')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created new directory {}".format(data_dir))

    if (os.path.exists(zip_path)):
        print("Zip file was downloaded and extracted before!")
    else:
        if (os.path.exists(os.path.join(data_dir, project_name, 'download/'))):
            pass
        else:
            os.makedirs(os.path.join(data_dir, project_name, 'download/'))
            urllib.request.urlretrieve(zip_url, zip_path)
            print("Downloaded data as {}".format(zip_path))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train'),
                        os.path.join(data_dir, project_name, 'download/'))
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val'),
                        os.path.join(data_dir, project_name, 'download/'))
            if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test')):
                shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test'),
                        os.path.join(data_dir, project_name, 'download/'))
            print("Unzipped data to {}".format(os.path.join(data_dir, project_name, 'download/')))


def make_dirs(data_dir, project_name):
    """
        Makes directories - `train`, `val, `test` and subdirectories under each `images` and `masks`
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
    """
    image_path_train = os.path.join(data_dir, project_name, 'train', 'images/')
    instance_path_train = os.path.join(data_dir, project_name, 'train', 'masks/')
    image_path_val = os.path.join(data_dir, project_name, 'val', 'images/')
    instance_path_val = os.path.join(data_dir, project_name, 'val', 'masks/')
    image_path_test = os.path.join(data_dir, project_name, 'test', 'images/')
    instance_path_test = os.path.join(data_dir, project_name, 'test', 'masks/')

    if not os.path.exists(image_path_train):
        os.makedirs(os.path.dirname(image_path_train))
        print("Created new directory : {}".format(image_path_train))

    if not os.path.exists(instance_path_train):
        os.makedirs(os.path.dirname(instance_path_train))
        print("Created new directory : {}".format(instance_path_train))

    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))

    if not os.path.exists(image_path_test):
        os.makedirs(os.path.dirname(image_path_test))
        print("Created new directory : {}".format(image_path_test))

    if not os.path.exists(instance_path_test):
        os.makedirs(os.path.dirname(instance_path_test))
        print("Created new directory : {}".format(instance_path_test))

        
        
def split_train_crops(project_name, center, crops_dir = 'crops', subset=0.15, by_fraction = True, train_name = 'train', seed=1000):
    imageDir = os.path.join(crops_dir, project_name, train_name, 'images')
    instanceDir = os.path.join(crops_dir, project_name, train_name, 'masks')
    centerDir = os.path.join(crops_dir, project_name, train_name, 'center-'+center)
    
    image_names = sorted(glob(os.path.join(imageDir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instanceDir, '*.tif')))
    center_names = sorted(glob(os.path.join(centerDir, '*.tif')))
    
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if (by_fraction):
        subsetLen = int(subset * len(image_names))
    else:
        subsetLen = int(subset)

    valIndices = indices[:subsetLen]
    
    image_path_val = os.path.join(crops_dir, project_name, 'val', 'images/')
    instance_path_val = os.path.join(crops_dir, project_name, 'val', 'masks/')
    center_path_val = os.path.join(crops_dir, project_name, 'val', 'center-'+center+'/')
    
    val_images_exist = False
    val_masks_exist = False
    val_center_images_exist = False
    
    if not os.path.exists(image_path_val):
        os.makedirs(os.path.dirname(image_path_val))
        print("Created new directory : {}".format(image_path_val))
    else:
        val_images_exist=True

    if not os.path.exists(instance_path_val):
        os.makedirs(os.path.dirname(instance_path_val))
        print("Created new directory : {}".format(instance_path_val))
    else:
        val_masks_exist=True
    
    if not os.path.exists(center_path_val):
        os.makedirs(os.path.dirname(center_path_val))
        print("Created new directory : {}".format(center_path_val))
    else:
        val_center_images_exist=True
        
    if not val_images_exist and not val_masks_exist and not val_center_images_exist:
        for val_index in valIndices:
            shutil.move(image_names[val_index], os.path.join(crops_dir, project_name, 'val', 'images'))
            shutil.move(instance_names[val_index], os.path.join(crops_dir, project_name, 'val', 'masks'))
            shutil.move(center_names[val_index], os.path.join(crops_dir, project_name, 'val', 'center-'+center))
        
        print("Val Images/Masks/Center-{}-image crops saved at {}".format(center, os.path.join(crops_dir, project_name, 'val')))
    else:
        print("Val Images/Masks/Center-{}-image crops already available at {}".format(center, os.path.join(crops_dir, project_name, 'val')))

def split_train_val(data_dir, project_name, train_val_name, subset=0.15, by_fraction = True, seed=1000):
    """
        Splits the `train` directory into `val` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        train_val_name: string
            Indicates the name of the sub-directory under `project-name` which must be split
        subset: float
            Indicates the fraction of data to be reserved for validation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-val partitions.
    """

    imageDir = os.path.join(data_dir, project_name, 'download', train_val_name, 'images')
    instanceDir = os.path.join(data_dir, project_name, 'download', train_val_name, 'masks')
    image_names = sorted(glob(os.path.join(imageDir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instanceDir, '*.tif')))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subsetLen = int(subset * len(image_names))
    else:
        subsetLen = int(subset)
    valIndices = indices[:subsetLen]
    trainIndices = indices[subsetLen:]
    make_dirs(data_dir=data_dir, project_name=project_name)

    for val_index in valIndices:
        shutil.copy(image_names[val_index], os.path.join(data_dir, project_name, 'val', 'images'))
        shutil.copy(instance_names[val_index], os.path.join(data_dir, project_name, 'val', 'masks'))

    for trainIndex in trainIndices:
        shutil.copy(image_names[trainIndex], os.path.join(data_dir, project_name, 'train', 'images'))
        shutil.copy(instance_names[trainIndex], os.path.join(data_dir, project_name, 'train', 'masks'))

    imageDir = os.path.join(data_dir, project_name, 'download', 'test', 'images')
    instanceDir = os.path.join(data_dir, project_name, 'download', 'test', 'masks')
    image_names = sorted(glob(os.path.join(imageDir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instanceDir, '*.tif')))
    testIndices = np.arange(len(image_names))
    for testIndex in testIndices:
        shutil.copy(image_names[testIndex], os.path.join(data_dir, project_name, 'test', 'images'))
        shutil.copy(instance_names[testIndex], os.path.join(data_dir, project_name, 'test', 'masks'))
    print("Train-Val-Test Images/Masks copied to {}".format(os.path.join(data_dir, project_name)))


def split_train_test(data_dir, project_name, train_test_name, subset=0.5, by_fraction = True, seed=1000):
    """
        Splits the `train` directory into `test` directory using the partition percentage of `subset`.
        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        train_test_name: string
            Indicates the name of the sub-directory under `project-name` which must be split
        subset: float
            Indicates the fraction of data to be reserved for evaluation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-test partitions.
    """

    image_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'images')
    instance_dir = os.path.join(data_dir, project_name, 'download', train_test_name, 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
    indices = np.arange(len(image_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subsetLen = int(subset * len(image_names))
    else:
        subsetLen = int(subset)
    test_indices = indices[:subsetLen]
    #make_dirs(data_dir=data_dir, project_name=project_name)
    test_images_exist = False
    test_masks_exist = False
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'images')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'images'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'images')))
    else:
        test_images_exist= True
    if not os.path.exists(os.path.join(data_dir, project_name, 'download', 'test', 'masks')):
        os.makedirs(os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Created new directory : {}".format(os.path.join(data_dir, project_name, 'download', 'test', 'masks')))
    else:
        test_masks_exist= True
    if not test_images_exist and not test_masks_exist:
        for test_index in test_indices:
            shutil.move(image_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'images'))
            shutil.move(instance_names[test_index], os.path.join(data_dir, project_name, 'download', 'test', 'masks'))
        print("Train-Test Images/Masks saved at {}".format(os.path.join(data_dir, project_name, 'download')))
    else:
        print("Train-Test Images/Masks already available at {}".format(os.path.join(data_dir, project_name, 'download')))
    

def calculate_foreground_weight(data_dir, project_name, train_val_name, mode, one_hot):
    instance_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'masks')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

    statistics = []
    for i in tqdm(range(len(instance_names))):
        ma = tifffile.imread(instance_names[i])
        if (mode == '2d'):
            statistics.append(10.0)
        elif (mode =='3d'):
            z, y, x = np.where(ma == 0)
            len_bg = len(z)
            z, y, x = np.where(ma > 0)
            len_fg = len(z)
            statistics.append(len_bg / len_fg)
    print("Foreground weight of the `{}` dataset set equal to {:.3f}".format(project_name, np.mean(statistics)))
    return np.mean(statistics)

def calculate_min_object_size(data_dir, project_name, train_val_name, mode, one_hot):
    instance_names = []
    size_list=[]
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'masks')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))
    
    for i in tqdm(range(len(instance_names))):
        ma = tifffile.imread(instance_names[i])
        if (one_hot and mode=='2d'):
            for z in range(ma.shape[0]):
                y,x=np.where(ma[z]==1)
                size_list.append(len(y))
        elif(not one_hot and mode=='2d'):
            ids = np.unique(ma)
            ids = ids[ids!=0]
            for id in ids:
                y,x = np.where(ma == id)
                size_list.append(len(y))
        elif(not one_hot and mode=='3d'):
            ids = np.unique(ma)
            ids = ids[ids!=0]
            for id in ids:
                z,y,x = np.where(ma == id)
                size_list.append(len(z))
    print("Minimum object size of the `{}` dataset is equal to {}".format(project_name, np.min(size_list)))
    return np.min(size_list).astype(np.float)

def calculate_max_eval_image_size(data_dir, project_name, test_name, mode, one_hot):
    instance_names = []
    size_z_list= []
    size_y_list = []
    size_x_list = []
    for name in test_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'masks')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

    for i in tqdm(range(len(instance_names))):
        ma = tifffile.imread(instance_names[i])
        if (one_hot and mode == '2d'):
            size_y_list.append(ma.shape[1])
            size_x_list.append(ma.shape[2])
        elif (not one_hot and mode == '2d'):
            size_y_list.append(ma.shape[0])
            size_x_list.append(ma.shape[1])
        elif (not one_hot and mode == '3d'):
            size_z_list.append(ma.shape[0])
            size_y_list.append(ma.shape[1])
            size_x_list.append(ma.shape[2])
    if mode == '2d':
        max_y = np.max(size_y_list)
        max_x = np.max(size_x_list)
        if(max_y % 8 !=0):
            n = max_y //8
            max_y = np.clip((n+1)*8, a_min=1024, a_max=None)
        else:
            max_y = np.clip(max_y, a_min=1024, a_max=None)
        if (max_x % 8 != 0):
            n = max_x // 8
            max_x = np.clip((n + 1) * 8, a_min=1024, a_max=None)
        else:
            max_x = np.clip(max_x, a_min=1024, a_max = None)
        max_x_y = np.maximum(max_x, max_y)
        max_x = max_x_y
        max_y = max_x_y
        print("Maximum evaluation image size of the `{}` dataset set equal to ({}, {})".format(project_name, max_y, max_x))
        return None, max_y.astype(np.float), max_x.astype(np.float)
    elif mode == '3d':
        max_z = np.max(size_z_list)
        max_y = np.max(size_y_list)
        max_x = np.max(size_x_list)
        if(max_z % 8 !=0):
            n = max_z //8
            max_z = (n+1)*8
        if(max_y % 8 !=0):
            n = max_y //8
            max_y = (n+1)*8
        if (max_x % 8 != 0):
            n = max_x // 8
            max_x = (n + 1)*8
        max_x_y = np.maximum(max_x, max_y)
        max_x = max_x_y
        max_y = max_x_y
        print("Maximum evaluation image size of the `{}` dataset set equal to  (n_z = {}, n_y = {}, n_x = {})".format(project_name, max_z, max_y, max_x))
        return max_z.astype(np.float), max_y.astype(np.float), max_x.astype(np.float)

def calculate_avg_background_intensity(data_dir, project_name, train_val_name, one_hot):
    instance_names = []
    image_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'masks')
        image_dir = os.path.join(data_dir, project_name, name, 'images')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))
        image_names +=sorted(glob(os.path.join(image_dir, '*.tif')))
    statistics = []
    if one_hot:
        for i in tqdm(range(len(instance_names))):
            ma = tifffile.imread(instance_names[i])
            bg_mask = ma == 0
            im = tifffile.imread(image_names[i])
            statistics.append(np.average(im[np.min(bg_mask, 0)]))
    else:
        for i in tqdm(range(len(instance_names))):
            ma = tifffile.imread(instance_names[i])
            bg_mask = ma == 0
            im = tifffile.imread(image_names[i])
            statistics.append(np.average(im[bg_mask]))
    print("Average background intensity of the `{}` dataset set equal to {:.3f}".format(project_name, np.mean(statistics)))
    return np.mean(statistics)

def get_data_properties(data_dir, project_name, train_val_name, test_name, mode, one_hot):
    data_properties_dir = {}
    data_properties_dir['foreground_weight'] = calculate_foreground_weight(data_dir, project_name, train_val_name, mode, one_hot)
    data_properties_dir['min_object_size'] = calculate_min_object_size(data_dir, project_name, train_val_name, mode, one_hot).astype(np.float)
    data_properties_dir['n_z'], data_properties_dir['n_y'], data_properties_dir['n_x'] =  calculate_max_eval_image_size(data_dir, project_name, test_name, mode, one_hot)
    data_properties_dir['one_hot']=one_hot
    data_properties_dir['avg_background_intensity'] = calculate_avg_background_intensity(data_dir, project_name, train_val_name, one_hot)
    return data_properties_dir
