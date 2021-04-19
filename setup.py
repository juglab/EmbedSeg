from __future__ import absolute_import
import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EmbedSeg", 
    version="0.2.1",
    author="Manan Lalit, Pavel Tomancak, Florian Jug",
    author_email="lalit@mpi-cbg.de",
    description="EmbedSeg provides automatic detection and segmentation of objects in microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juglab/EmbedSeg/",
    project_urls={
        "Bug Tracker": "https://github.com/juglab/EmbedSeg/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          "tifffile",
          "numba",
          "tqdm",
          "jupyter",
          "pandas",
          "seaborn",
          "scikit-image",
          "colorspacious",
          "itkwidgets"
        ]
)

