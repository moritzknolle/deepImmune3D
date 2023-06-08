# deepImmune3D [![DOI](https://zenodo.org/badge/478273214.svg)](https://zenodo.org/badge/latestdoi/478273214)
- predict melanoma tumour immune status from confocal, volumetric immunofluorescence microscopy images
- model interpretability with occlusion sensitivity and gradient-based attribution maps (smoothGrad)

## Prediction Pipeline overview
![Alt text](pipeline.png?raw=true "Title")

### Install required packages
simply run: `conda env create --file env.yml`

### Example workflow
For an example workflow on how to train a deep learning model on volumetric images and subsequently obtain interpretability maps see `example.ipynb`. Note that running the same workflow on the original, high-resolution con-focal microscopy images necessitates a GPU with > 40 GB of memory.

### Linear model for immune status prediction
As demonstrated in the paper (Figure 1H) it is also possible to obtain a high-performing (linear) model from hand-crafed marker-specific cell-count features. For an example on how to construct such a model see `linear/cell_count.py`. Note that this approach does not necessitate the use of a GPU.
