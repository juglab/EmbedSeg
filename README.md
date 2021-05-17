[![License](https://img.shields.io/pypi/l/EmbedSeg.svg?color=green)](https://github.com/juglab/EmbedSeg/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/EmbedSeg.svg?color=green)](https://pypi.org/project/EmbedSeg)

<p align="center">
  <img src="https://user-images.githubusercontent.com/34229641/117163211-bb727300-adc3-11eb-8fe4-ebd7d0e5fceb.png" width=300 />
</p>
<h2 align="center">Embedding-based Instance Segmentation in Microscopy</h2>

## Table of Contents

- **[Introduction](#introduction)**
- **[Dependencies](#dependencies)**
- **[Getting Started](#getting-started)**
- **[Datasets](#datasets)**
- **[Training & Inference on your data](#training-and-inference-on-your-data)**
- **[Animated figures](#animated-figures)**
- **[Contributing](#contributing)**
- **[Issues](#issues)**
- **[Citation](#citation)**
- **[Acknowledgements](#acknowledgements)**


### Introduction
This repository hosts the version of the code used for the **[preprint](https://arxiv.org/abs/2101.10033)** **Embedding-based Instance Segmentation of Microscopy Images**. 

We refer to the techniques elaborated in the publication, here as **EmbedSeg**. `EmbedSeg` is a method to perform instance-segmentation of objects in microscopy images, based on the ideas by **[Neven et al, 2019](https://arxiv.org/abs/1906.11109)**. 

With `EmbedSeg`, we obtain state-of-the-art results on multiple real-world microscopy datasets. `EmbedSeg` has a small enough memory footprint (between 0.7 to about 3 GB) to allow network training on virtually all CUDA enabled hardware, including laptops.


### Dependencies 
We have tested this implementation using `pytorch` version 1.1.0 and `cudatoolkit` version 10.0 on a `linux` OS machine. 

- One could install `EmbedSeg` with **[pip]**:
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


### Datasets
3D datasets are available as release assets **[here](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0)**. 
![datasets](https://user-images.githubusercontent.com/34229641/118558022-5bd27b00-b766-11eb-889b-00886b725c2a.png)

### Training and Inference on your data
   
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

### Animated Figures

<p align="center">
  <img src="https://mlbyml.github.io/EmbedSeg_RC/images/teaser/train_images_painted.gif" alt="teaser" width="500"/>
</p>


### Contributing

Contributions are very welcome. Tests can be run with **[tox]**.

### Issues

If you encounter any problems, please **[file an issue]** along with a detailed description.

[file an issue]: https://github.com/juglab/EmbedSeg/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/EmbedSeg/


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
### Acknowledgements
The authors would like to thank the Scientific Computing Facility at MPI-CBG, thank Matthias Arzt,  Joran  Deschamps  and  Nuno  Pimpao  Martins  for  feedback  and  testing.    Alf  Honigmann and  Anna  Goncharova  provided  the  `Mouse-Organoid-Cells-CBG`  data  and  annotations.   Jacqueline Tabler and Diana Afonso provided the `Mouse-Skull-Nuclei-CBG` dataset and annotations.  This work was supported by the German Federal Ministry of Research and Education (BMBF) under the codes 031L0102 (de.NBI) and 01IS18026C (ScaDS2), and the German Research Foundation (DFG) under the code JU3110/1-1(FiSS) and TO563/8-1 (FiSS). P.T. was supported by the European Regional Development Fund in the IT4Innovations national supercomputing center,  project number CZ.02.1.01/0.0/0.0/16013/0001791 within the Program Research, Development and Education.
