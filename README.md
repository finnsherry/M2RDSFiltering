# M2RDSFiltering
This repository contains code to perform Regularised Diffusion-Shock (RDS) Filtering on two dimensional Euclidean space, $\mathbb{R}^2$, and the corresponding space of positions and orientations, $\mathbb{M}_2 \coloneqq \mathbb{R}^2 \times S^1$, as described in [[1]](#1). The results of [[1]](#1) can be reproduced with the notebooks in the `experiments` directory.

This work extends $\mathbb{R}^2$ RDS by Schaefer & Weickert [[2]](#2).

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
| `dsfilter.DS_enhancing_LI`    | left-invariant $\mathbb{M}_2$ RDS denoising [[1]](#1) |
| `dsfilter.DS_enhancing_gauge` | gauge frame $\mathbb{M}_2$ RDS denoising [[1]](#1) |
| `dsfilter.DS_enhancing_R2`    | $\mathbb{R}^2$ RDS denoising [[2]](#2) |
| `dsfilter.TV_enhancing_LI`    | left-invariant $\mathbb{M}_2$ T(R-T)V denoising [[3]](#3)[[4]](#4) |
| `dsfilter.TV_enhancing_gauge` | gauge frame $\mathbb{M}_2$ T(R-T)V denoising [[3]](#3)[[4]](#4) |
| `dsfilter.DS_inpainting_LI`   | left-invariant $\mathbb{M}_2$ RDS inpainting [[1]](#1) |
| `dsfilter.DS_inpainting_R2`   | $\mathbb{R}^2$ RDS inpainting [[2]](#2) |

## Cite
If you use this code in your own work, please cite our paper:

<a id="1">[1]</a> Sherry, F.M., Schaefer, K., Duits, R. "Diffusion-Shock Filtering on the Space of Positions and Orientations." 10th International Conference on Scale Space and Variational Methods in Computer Vision (SSVM) (2025). https://doi.org/10.1007/978-3-031-92369-2_16
```
@inproceedings{Sherry2025DSM2,
  author =       {Sherry, Finn M. and Schaefer, Kristina and Duits, Remco},
  title =        {{Diffusion-Shock Filtering on the Space of Positions and Orientations}},
  booktitle =    {10th International Conference on Scale Space and Variational Methods in Computer Vision},
  publisher =    {Springer},
  year =         {2025},
  address =      {Totnes, United Kingdom},
  pages =        {205--217},
  doi =          {10.1007/978-3-031-92369-2_16},
  editor =       {Bubba, Tatiana and Papafitsoros, Kostas and Gazzola, Silvia and Pereyra, Marcelo and Gaburro, Romina and Schönlieb, Carola}
}
```

We extend RDS filtering on $\mathbb{R}^2$ by Schaefer & Weickert to $\mathbb{M}_2$:

<a id="2">[2]</a> Schaefer, K., Weickert, J. "Regularised Diffusion-Shock Inpainting." Journal of Mathematical Imaging and Vision (2024). https://doi.org/10.1007/s10851-024-01175-0
```
@article{Schaefer2024RDS,
  author =       {Schaefer, Kristina and Weickert, Joachim},
  title =        {{Regularised Diffusion-Shock Inpainting}},
  journal =      {Journal of Mathematical Imaging and Vision},
  publisher =    {Springer},
  year =         {2024},
  volume =       {66},
  pages =        {447--463},
  doi =          {10.1007/s10851-024-01175-0}
}
```

We compare to Total Roto-Translational Variation (TR-TV) flow by Chambolle & Pock and Smets et al.; this repository contains our own reimplementations:

<a id="3">[3]</a> Chambolle, A., Pock, Th. "Total roto-translational variation." Numerische Mathematik (2019). https://doi.org/10.1007/s00211-019-01026-w
```
@article{Chambolle2019TRTV,
  author =       {Chambolle, Antonin and Pock, Thomas},
  title =        {{Total roto-translational variation}},
  journal =      {Numerische Mathematik},
  publisher =    {Springer},
  year =         {2019},
  volume =       {142},
  pages =        {611--666},
  doi =          {10.1007/s00211-019-01026-w}
}
```

<a id="4">[4]</a> Smets, B.M.N., Portegies, J.W., St-Onge, E., Duits, R. "Total Variation and Mean Curvature PDEs on the Homogeneous Space of Positions and Orientations." Journal of Mathematical Imaging and Vision (2021). https://doi.org/10.1007/s10851-020-00991-4
```
@article{Smets2021TV,
  author =       {Smets, Bart M.N. and Portegies, Jacobus W. and St-Onge, Etienne and Duits, Remco},
  title =        {{Total Variation and Mean Curvature PDEs on the Homogeneous Space of Positions and Orientations}},
  journal =      {Journal of Mathematical Imaging and Vision},
  publisher =    {Springer},
  year =         {2021},
  volume =       {63},
  pages =        {237--262},
  doi =          {10.1007/s10851-020-00991-4}
}
```