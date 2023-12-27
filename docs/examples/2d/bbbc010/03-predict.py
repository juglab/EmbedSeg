# # Predict

import json
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from EmbedSeg.test import begin_evaluating
from EmbedSeg.train import invert_one_hot
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.utils.visualize import visualize
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")


# ## Specify the path to the evaluation images

data_dir = "../../../data"
project_name = "bbbc010-2012"
print(f"Evaluation images shall be read from: {Path(data_dir)/project_name}.")

# ## Specify evaluation parameters

# Some hints:
# * `tta`: Setting this to True (default) would enable **test-time augmentation**
# * `ap_val`: This parameter ("average precision value") comes into action if ground truth segmentations exist for evaluation images, and allows to compare how good our predictions are versus the available ground truth segmentations.
# * `checkpoint_path`: This parameter provides the path to the trained model weights which you would like to use for evaluation. One could test the pretrained model to get a quick glimpse on the results.
# * `save_dir`: This parameter specifies the path to the prediction instances. Equal to `inference` by default.
#
# In the cell after this one, a `test_configs` dictionary is generated from the parameters specified here!
# <a id='checkpoint'></a>

# +
# uncomment for the model trained by you
# checkpoint_path = Path('experiment')/str(project_name+'-'+'demo')/'best_iou_model.pth'
# if Path('data_properties.json').is_file():
#     with open(Path('data_properties.json')) as json_file:
#         data = json.load(json_file)
#         one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(data['min_object_size']), int(data['n_y']), int(data['n_x']), float(data['avg_background_intensity'])
# if Path('normalization.json').is_file():
#     with open(Path('normalization.json')) as json_file:
#         data = json.load(json_file)
#         norm = data['norm']

# use the following for the pretrained model weights
torch.hub.download_url_to_file(
    url="https://owncloud.mpi-cbg.de/index.php/s/yTNnewbAEFx4qBJ/download",
    dst="pretrained_model",
    progress=True,
)


with zipfile.ZipFile("pretrained_model", "r") as zip_ref:
    zip_ref.extractall("")
checkpoint_path = Path(str(project_name + "-" + "demo")) / "best_iou_model.pth"
if (Path(str(project_name + "-" + "demo")) / "data_properties.json").is_file():
    with open(
        Path(str(project_name + "-" + "demo")) / "data_properties.json"
    ) as json_file:
        data = json.load(json_file)
        one_hot, data_type, min_object_size, n_y, n_x, avg_bg = (
            data["one_hot"],
            data["data_type"],
            int(data["min_object_size"]),
            int(data["n_y"]),
            int(data["n_x"]),
            float(data["avg_background_intensity"]),
        )
if (Path(str(project_name + "-" + "demo")) / "normalization.json").is_file():
    with open(
        Path(str(project_name + "-" + "demo")) / "normalization.json"
    ) as json_file:
        data = json.load(json_file)
        norm = data["norm"]

# -

tta = True
ap_val = 0.5
save_dir = "./inference"

if Path(checkpoint_path).exists():
    print(f"Trained model weights found at : {checkpoint_path}")
else:
    print("Trained model weights were not found at the specified location!")

# ## Create `test_configs` dictionary from the above-specified parameters

# Setting **`cluster_fast=False`** would produce more cell detections but the inference process would take longer.<br>
# <b>Set device to be the same as the one used during training.</b> It can be set as one of `cpu`, `mps` or `cuda:n` (for example, device = 'cuda:0')

test_configs = create_test_configs_dict(
    data_dir=Path(data_dir) / project_name,
    checkpoint_path=checkpoint_path,
    tta=tta,
    ap_val=ap_val,
    min_object_size=min_object_size,
    save_dir=save_dir,
    norm=norm,
    data_type=data_type,
    one_hot=one_hot,
    n_y=n_y,
    n_x=n_x,
    cluster_fast=True,
    device="cuda:0",
)

# ## Begin Evaluating

# Setting `verbose` to True shows you Average Precision at IOU threshold specified by `ap_val` above for each individual image. The higher this score is, the better the network has learnt to perform instance segmentation on these unseen images.

begin_evaluating(test_configs, verbose=False)

# <div class="alert alert-block alert-warning">
#   Common causes for a low score/error is: <br>
#     1. Accessing the model weights at the wrong location: simply editing the <b> checkpoint_path</b> would fix the issue.  <br>
#     2. GPU is out of memory - ensure that you shutdown <i>02-train.ipynb</i> notebook
# </div>

# ## Load a glasbey-style color map

urllib.request.urlretrieve(
    "https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/cmap_60.npy",
    "cmap_60.npy",
)
new_cmp = ListedColormap(np.load("cmap_60.npy"))

# ## Investigate some qualitative results

# Here you can investigate some quantitative predictions. GT segmentations and predictions, if they exist, are loaded from sub-directories under `save_dir`.
# Simply change `index` in the next two cells, to show the prediction for a random index.
# Going clockwise from top-left is
#
#     * the raw-image which needs to be segmented,
#     * the corresponding ground truth instance mask,
#     * the network predicted instance mask, and
#     * the confidence map predicted by the model

# %matplotlib inline
prediction_file_names = sorted(list((Path(save_dir) / "predictions").iterdir()))
ground_truth_file_names = sorted(list((Path(save_dir) / "ground-truth").iterdir()))
seed_file_names = sorted(list((Path(save_dir) / "seeds").iterdir()))
image_file_names = sorted(
    list((Path(data_dir) / project_name / "test" / "images").iterdir())
)

index = 24
print(f"Image filename is {Path(image_file_names[index]).name} and index is {index}.")
prediction = tifffile.imread(prediction_file_names[index])
image = tifffile.imread(image_file_names[index])
seed = tifffile.imread(seed_file_names[index])
if len(ground_truth_file_names) > 0:
    ground_truth = tifffile.imread(ground_truth_file_names[index])
    visualize(
        image=image,
        prediction=prediction,
        ground_truth=invert_one_hot(ground_truth),
        seed=seed,
        new_cmp=new_cmp,
    )
else:
    visualize(
        image=image,
        prediction=prediction,
        ground_truth=None,
        seed=seed,
        new_cmp=new_cmp,
    )

index = 23
print(f"Image filename is {Path(image_file_names[index]).name} and index is {index}.")
prediction = tifffile.imread(prediction_file_names[index])
image = tifffile.imread(image_file_names[index])
seed = tifffile.imread(seed_file_names[index])
if len(ground_truth_file_names) > 0:
    ground_truth = tifffile.imread(ground_truth_file_names[index])
    visualize(
        image=image,
        prediction=prediction,
        ground_truth=invert_one_hot(ground_truth),
        seed=seed,
        new_cmp=new_cmp,
    )
else:
    visualize(
        image=image,
        prediction=prediction,
        ground_truth=None,
        seed=seed,
        new_cmp=new_cmp,
    )
