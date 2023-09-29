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
- **[Issues](#issues)**
- **[Citation](#citation)**
- **[Acknowledgements](#acknowledgements)**


### Introduction
This repository hosts the version of the code used for the **[publication](https://proceedings.mlr.press/v143/lalit21a.html)** **Embedding-based Instance Segmentation of Microscopy Images**. 

We refer to the techniques elaborated in the publication, here as **EmbedSeg**. `EmbedSeg` is a method to perform instance-segmentation of objects in microscopy images, based on the ideas by **[Neven et al, 2019](https://arxiv.org/abs/1906.11109)**. 

With `EmbedSeg`, we obtain state-of-the-art results on multiple real-world microscopy datasets. `EmbedSeg` has a small enough memory footprint (between 0.7 to about 3 GB) to allow network training on virtually all CUDA enabled hardware, including laptops.


### Dependencies 

One could execute these lines of code to run this branch with GPU support:

```
mamba create -n EmbedSeg python
mamba activate EmbedSeg
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/juglab/EmbedSeg.git
cd EmbedSeg
pip install -e .
```

For CPU support, one could execute the following lines of code:


```
mamba create -n EmbedSeg python
mamba activate EmbedSeg
pip install torch torchvision
git clone https://github.com/juglab/EmbedSeg.git
cd EmbedSeg
```

### Getting Started

Look in the `examples` directory,  and try out the `DSB-2018` notebooks for 2D images or `Mouse-Organoid-Cells-CBG` notebooks for volumetric (3D) images. Please make sure to select `Kernel > Change kernel` to `EmbedSegEnv`.   


### Datasets
3D datasets are available as release assets **[here](https://github.com/juglab/EmbedSeg/releases/tag/v0.1.0)**. 
![datasets](https://user-images.githubusercontent.com/34229641/118558022-5bd27b00-b766-11eb-889b-00886b725c2a.png)

### Training and Inference on your data
   
`*.tif`-type images and the corresponding masks should be respectively present under `images` and `masks`, under directories `train`, `val` and `test`. (In order to prepare such instance masks, one could use the Fiji plugin <b>Labkit</b> as suggested <b>[here](https://github.com/juglab/EmbedSeg/wiki/01---Use-Labkit-to-prepare-instance-masks)</b>). The following would be a desired structure as to how data should be prepared.

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

### Issues

If you encounter any problems, please **[file an issue]** along with a detailed description.

[file an issue]: https://github.com/juglab/EmbedSeg/issues


### Citation

If you find our work useful in your research, please consider citing:

```bibtex
@InProceedings{lalit2021embedseg,
  title = 	 {Embedding-based Instance Segmentation in Microscopy},
  author =       {Lalit, Manan and Tomancak, Pavel and Jug, Florian},
  booktitle = 	 {Proceedings of the Fourth Conference on Medical Imaging with Deep Learning},
  pages = 	 {399--415},
  year = 	 {2021},
  editor = 	 {Heinrich, Mattias and Dou, Qi and de Bruijne, Marleen and Lellmann, Jan and Schläfer, Alexander and Ernst, Floris},
  volume = 	 {143},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--09 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v143/lalit21a/lalit21a.pdf},
  url = 	 {https://proceedings.mlr.press/v143/lalit21a.html},
}
```

and 

```bibtex
@article{lalit2022mia,
title = {EmbedSeg: Embedding-based Instance Segmentation for Biomedical Microscopy Data},
journal = {Medical Image Analysis},
volume = {81},
pages = {102523},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102523},
url = {https://www.sciencedirect.com/science/article/pii/S1361841522001700},
author = {Manan Lalit and Pavel Tomancak and Florian Jug},
}
```


### Acknowledgements

The authors would like to thank the Scientific Computing Facility at MPI-CBG, thank Matthias Arzt,  Joran  Deschamps  and  Nuno  Pimpao  Martins  for  feedback  and  testing.    Alf  Honigmann and  Anna  Goncharova  provided  the  `Mouse-Organoid-Cells-CBG`  data  and  annotations.   Jacqueline Tabler and Diana Afonso provided the `Mouse-Skull-Nuclei-CBG` dataset and annotations.  This work was supported by the German Federal Ministry of Research and Education (BMBF) under the codes 031L0102 (de.NBI) and 01IS18026C (ScaDS2), and the German Research Foundation (DFG) under the code JU3110/1-1(FiSS) and TO563/8-1 (FiSS). P.T. was supported by the European Regional Development Fund in the IT4Innovations national supercomputing center,  project number CZ.02.1.01/0.0/0.0/16013/0001791 within the Program Research, Development and Education.

The authors would like to thank the authors of **[StarDist](https://github.com/stardist/stardist)** repository for several useful, helper functions. The authors would also like to thank Sahar Kakavand and Marco Dalla Vecchia for feedback on notebooks. 
