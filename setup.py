import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EmbedSeg", # Replace with your own username
    version="0.0.1",
    author="Manan Lalit",
    author_email="lalit@mpi-cbg.de",
    description="EmbedSeg provides automatic detection and segmentation of objects in microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juglab/EmbedSeg/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

