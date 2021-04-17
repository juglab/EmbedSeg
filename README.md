## EmbedSeg 

### Introduction
This repository hosts the version of the code used for the **[preprint](https://arxiv.org/abs/2101.10033)** **Embedding-based Instance Segmentation of Microscopy Images**. For a short summary of the main attributes of the publication, please check out the **[project webpage](https://juglab.github.io/EmbedSeg/)**.

We refer to the techniques elaborated in the publication, here as **EmbedSeg**. `EmbedSeg` is a method to perform instance-segmentation of objects in microscopy images, based on the ideas by **[Neven et al, 2019](https://arxiv.org/abs/1906.11109)**. 

<p align="center">
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/train_images_painted.gif" alt="teaser" width="500"/>
</p>


With `EmbedSeg`, we obtain state-of-the-art results on multiple real-world microscopy datasets. `EmbedSeg` has a small enough memory footprint (between 0.7 to about 3 GB) to allow network training on virtually all CUDA enabled hardware, including laptops.

### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@misc{lalit2021embeddingbased,
      title={Embedding-based Instance Segmentation of Microscopy Images}, 
      author={Manan Lalit and Pavel Tomancak and Florian Jug},
      year={2021},
      eprint={2101.10033},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

### Dependencies 
We have tested this implementation using `pytorch` version 1.1.0 and `cudatoolkit` version 10.0 on a `linux` OS machine. 

- One could install `EmbedSeg` with `pip`:
```
conda create -n EmbedSegEnv python==3.7
conda activate EmbedSegEnv
python3 -m pip install EmbedSeg
```

and then install <b>[pytorch](https://pytorch.org/get-started/previous-versions/)</b>:
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
```

- Alternately, one could use the `environment.yml` file (this would also install `pytorch`, `torchvision` and `cudatoolkit`). 
Create a new environment using :

```conda env create -f path/to/environment.yml```.


### Getting Started

Look in the `examples` directory,  and try out one of the provided notebooks. Please make sure to select `Kernel > Change kernel` to `EmbedSegEnv`.   


### Training & Inference on your data
   
`*.tif`-type images and the corresponding masks should be respectively present under `images` and `masks`, under directories `train`, `val` and `test`. (In order to prepare such instance masks, one could use the Fiji plugin <b>Labkit</b> as suggested <b>[here](https://github.com/juglab/EmbedSeg/wiki/Use-Labkit-to-prepare-instance-masks)</b>). The following would be a desired structure as to how data should be prepared.

```
$data_dir
└───$project-name
    |───train
        └───images
            └───X0.tif
            └───...
            └───Xn.tif
        └───masks
            └───Y0.tif
            └───...
            └───Yn.tif
    |───val
        └───images
            └───...
        └───masks
            └───...
    |───test
        └───images
            └───...
        └───masks
            └───...
```


