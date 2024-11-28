# M2RDSFiltering
This repository contains code to perform Regularised Diffusion-Shock (RDS) Filtering on $\mathbb{R}^2$ and $\mathbb{M}_2$, as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the notebooks in the `experiments` directory.

## Installation
The core functionality of this repository requires:
* `python>=3.10`
* `taichi==1.6`
* `numpy`
* `scipy`
* `matplotlib`
* `tqdm`

To reproduce the experiments, one additionally needs:
* `jupyter`
* `pillow`
* `scikit-image`
* `bm3d`

Alternatively, one can create the required conda environment from `dsfilter.yml`:
```
conda env create -f dsfilter.yml
```
This creates a conda environment called `dsfilter`.

Subsequently, one must install the code of this project as a package, by running:
```
pip install -e .
```

## Functionality
The main functionality is exposed as top level functions:
| Method | Description |
| ------ | ----------- |
| `dsfilter.DS_enhancing_LI`    | left-invariant $\mathbb{M}_2$ RDS denoising |
| `dsfilter.DS_enhancing_gauge` | gauge frame $\mathbb{M}_2$ RDS denoising |
| `dsfilter.TV_enhancing_LI`    | left-invariant $\mathbb{M}_2$ T(R-T)V denoising |
| `dsfilter.TV_enhancing_gauge` | gauge frame $\mathbb{M}_2$ T(R-T)V denoising |
| `dsfilter.DS_inpainting_LI`   | left-invariant $\mathbb{M}_2$ RDS inpainting |
| `dsfilter.DS_inpainting_R2`   | $\mathbb{R}^2$ RDS inpainting |

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> Sherry, F.M., Schaefer, K., Duits, R. Diffusion-Shock Filtering on the Space of Positions and Orientations. 10th International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), (2025).

```
@inproceedings{Sherry2025DSM2,
  author =       {Sherry, Finn M. and Schaefer, Kristina and Duits, Remco},
  title =        {{Diffusion-Shock Filtering on the Space of Positions and Orientations}},
  booktitle =    {10th International Conference on Scale Space and Variational Methods in Computer Vision},
  publisher =    {Springer},
  year =         {2025},
  address =      {Totnes, United Kingdom},
  pages =        {},
  doi =          {},
  editor =       {}
}
```