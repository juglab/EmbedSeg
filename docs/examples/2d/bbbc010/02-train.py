# # Train

import json
import urllib.request
from pathlib import Path

import numpy as np
from EmbedSeg.train import begin_training
from EmbedSeg.utils.create_dicts import (
    create_configs,
    create_dataset_dict,
    create_loss_dict,
    create_model_dict,
)
from matplotlib.colors import ListedColormap

# ## Specify the path to `train`, `val` crops and the type of `center` embedding which we would like to train the network for:

# The train-val images, masks and center-images will be accessed from the path specified by `data_dir` and `project-name`.
# <a id='center'></a>

# +
data_dir = "crops"
project_name = "bbbc010-2012"
center = "medoid"

print(
    f"Project Name chosen as : {project_name}.",
    f"\nTrain-Val images-masks-center-images will be accessed from : {data_dir}.",
)
# -

try:
    assert center in {"medoid", "approximate-medoid", "centroid"}
    print(f"Spatial Embedding Location chosen as : {center}")
except AssertionError as e:
    e.args += (
        'Please specify center as one of : {"medoid", "approximate-medoid", "centroid"}',
        42,
    )
    raise

# ## Obtain properties of the dataset

# Here, we read the `dataset.json` file prepared in the `01-data` notebook previously.

if Path("data_properties.json").is_file():
    with open("data_properties.json") as json_file:
        data = json.load(json_file)
        one_hot, data_type, foreground_weight, n_y, n_x = (
            data["one_hot"],
            data["data_type"],
            int(data["foreground_weight"]),
            int(data["n_y"]),
            int(data["n_x"]),
        )

# ## Specify training dataset-related parameters

# Some hints:
# * The `train_size` attribute indicates the number of image-mask paired examples which the network would see in one complete epoch. Ideally this should be the number of `train` image crops. For the `bbbc010-2012` dataset, we obtain ~600 crops, but since we use a batch size of 1, we double the `train_size` to 1200 to give the model more time to converge.
#
# In the cell after this one, a `train_dataset_dict` dictionary is generated from the parameters specified here!

train_size = 2 * len(
    list((Path(data_dir) / project_name / "train" / "images").iterdir())
)
train_batch_size = 1

# ## Create the `train_dataset_dict` dictionary

train_dataset_dict = create_dataset_dict(
    data_dir=data_dir,
    project_name=project_name,
    center=center,
    size=train_size,
    batch_size=train_batch_size,
    one_hot=one_hot,
    type="train",
)

# ## Specify validation dataset-related parameters

# Some hints:
# * The size attribute indicates the number of image-mask paired examples which the network would see in one complete epoch. Here, it is recommended to set `val_size` equal to an integral multiple of total number of validation image crops. For example, for the `bbbc010-2012` dataset, we notice ~100 validation crops, but we set `val_size = 800` to obtain the optimal results across the 8-fold augmentation of these 100 crops. It would also be fine to set `val_size = 100`.
#
# In the cell after this one, a `val_dataset_dict` dictionary is generated from the parameters specified here!
#
#

val_size = 8 * len(list((Path(data_dir) / project_name / "val" / "images").iterdir()))
val_batch_size = 1

# ## Create the `val_dataset_dict` dictionary

val_dataset_dict = create_dataset_dict(
    data_dir=data_dir,
    project_name=project_name,
    center=center,
    size=val_size,
    batch_size=val_batch_size,
    one_hot=one_hot,
    type="val",
)

# ## Specify model-related parameters

# Some hints:
# * Set the `input_channels` attribute equal to the number of channels in the input images.
#
# In the cell after this one, a `model_dataset_dict` dictionary is generated from the parameters specified here!

input_channels = 1

# ## Create the `model_dict` dictionary

model_dict = create_model_dict(input_channels=input_channels)

# ## Create the `loss_dict` dictionary

loss_dict = create_loss_dict(foreground_weight=foreground_weight)

# ## Specify additional parameters

# Some hints:
# * The `n_epochs` attribute determines how long the training should proceed. In general for good results on `bbbbc_010` dataset with the configurations above, you should train for atleast 50 epochs. <b>But for best results, please train for 200 epochs</b>
# * The `save_dir` attribute identifies the location where the checkpoints and loss curve details are saved.
# * If one wishes to **resume training** from a previous checkpoint, they could point `resume_path` attribute appropriately. For example, one could set `resume_path = './experiment/bbbc010-2012-demo/checkpoint.pth'` to resume training from the last checkpoint.
# * The `one_hot` attribute should be set to True if the instance image is present in an one-hot encoded style (i.e. object instance is encoded as 1 in its own individual image slice) and False if the instance image is the same dimensions as the raw-image.
#
#
# In the cell after this one, a `configs` dictionary is generated from the parameters specified here!
# <a id='resume'></a>

n_epochs = 200
save_dir = Path("experiment") / str(project_name + "-demo")
resume_path = None

# ## Create the  `configs` dictionary

# <b>Set the `device` equal to the `cpu` or `mps` or `cuda:n` (for example, `device='cuda:0'`)</b>

configs = create_configs(
    n_epochs=n_epochs,
    one_hot=one_hot,
    resume_path=resume_path,
    save_dir=save_dir,
    n_y=n_y,
    n_x=n_x,
    device="cpu",
)

# ## Choose a `color map`

# Here, we load a `glasbey`-style color map. But other color maps such as `viridis`, `magma` etc would work equally well.

urllib.request.urlretrieve(
    "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/cmap_60.npy",
    "cmap_60.npy",
)
new_cmap = ListedColormap(np.load("cmap_60.npy"))

# ## Begin training!

# Executing the next cell would begin the training.

begin_training(
    train_dataset_dict,
    val_dataset_dict,
    model_dict,
    loss_dict,
    configs,
    color_map=new_cmap,
)

# <div class="alert alert-block alert-warning">
#   Common causes for errors during training, may include : <br>
#     1. Not having <b>center images</b> for  <b>both</b> train and val directories  <br>
#     2. <b>Mismatch</b> between type of center-images saved in <b>01-data.ipynb</b> and the type of center chosen in this notebook (see the <b><a href="#center"> center</a></b> parameter in the third code cell in this notebook)   <br>
#     3. In case of resuming training from a previous checkpoint, please ensure that the model weights are read from the correct directory, using the <b><a href="#resume"> resume_path</a></b> parameter. Additionally, please ensure that the <b>save_dir</b> parameter for saving the model weights points to a relevant directory.
# </div>
