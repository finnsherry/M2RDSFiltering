# M2RDSFiltering
Investigate Regularised Diffusion-Shock Filtering on $\mathbb{R}^2$ and $\mathbb{M}_2$.

## Dependencies
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

Alternatively, one can create the required conda environment from `min_env.yml`:
```
conda env create -f dsfilter.yml
```
This creates a conda environment called `dsfilter`.

Subsequently, one must install the code of this project as a package, by running:
```
pip install -e .
```

## Cite
If you use this code in your own work, please cite our paper:

Sherry, F.M., Schaefer, K., Duits, R. Diffusion-Shock Filtering on the Space of Positions and Orientations. 10th International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), (2025).

```
@inproceedings{AVK93,
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