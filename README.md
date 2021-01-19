## EmbedSeg 

### Introduction
This repository hosts the version of the code used for the **[publication]()** **Embedding-based Instance Segmentation of Microscopy Images**. For a short summary of the main attributes of the publication, please check out the **[project webpage]()**.

We refer to the techniques elaborated in the publication, here as **EmbedSeg**. `EmbedSeg` is a method to perform instance-segmentation of objects in microscopy images, and extends the formulation of **[Neven et al, 2019](https://arxiv.org/abs/1906.11109)**. 

<p float="left">
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_image_painted.gif" width="100" />
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_GT_painted.gif" width="100" /> 
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/X_9_painted.gif" width="100" />
</p>


In `EmbedSeg`, we suggest two simple tricks: by embedding interior pixels to instance medoids instead of the instance centroids and by including test-time augmentation during inference, we obtain state-of-the-art results on several datasets. Additionally by accumulating gradients over multiple steps, we allow our overall pipeline to have a small enough memory footprint to allow network training on virtually all CUDA enabled laptop hardware.

### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{2021:EmbedSeg,
  title={Embedding-based Instance Segmentation of Microscopy Images},
  author={Lalit, Manan and Tomancak, Pavel and Jug, Florian},
  journal={},
  year={2021}
}
```

### Dependencies 
We have tested this implementation using `pytorch` version 1.1.0 and `cudatoolkit` version 10.0 on a `linux` OS machine. 

In order to replicate results mentioned in the publication, one could use the same virtual environment (`EmbedSeg_environment.yml`) as used by us. Create a new environment, for example,  by entering the python command in the terminal `conda env create -f path/to/EmbedSeg_environment.yml`.

### Getting Started

Please open a new terminal window and run the following commands one after the other.

```shell
git clone https://github.com/juglab/EmbedSeg.git
cd EmbedSeg
git checkout v0.1
conda env create -f EmbedSeg_environment.yml
conda activate EmbedSegEnv
python3 -m pip install -e .
python3 -m ipykernel install --user --name EmbedSegEnv --display-name "EmbedSegEnv"
cd examples/2d
jupyter notebook
```

Look in the `examples` directory,  and try out one (or all) of the three sets of notebooks (`dsb-2018`, `usiigaci-2017` or `bbbc010-2012`). Please make sure to select `Kernel > Change kernel` to `EmbedSegEnv`. 


### Training & Inference on your data
   
`*.tif`-type images and the corresponding masks should be respectively present under `images` and `masks`, under directories `train`, `val` and `test`. These are cropped in smaller patches in the notebook `01-data.ipynb`. The following would be a desired structure as to how data should be prepared.

```
data
└───data-name
    |───train
	└───images
		└───X0.tif
		└───X1.tif
		└───...
		└───Xn.tif
	└───masks
		└───Y0.tif
		└───Y1.tif
		└───...
		└───Yn.tif
    |───val
	└───images
	└───masks
    |───test
	└───images
	└───masks
```


