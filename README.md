# Hyper-Transforming Latent Diffusion Models
[arXiv](https://arxiv.org/abs/2504.16580) | [BibTeX](#bibtex)
---


<p align="center">
<img src=assets/ivae.png />
</p>

<p align="center">
<img src=assets/ldm.png />
</p>



[**Hyper-Transforming Latent Diffusion Models**](https://www.arxiv.org/abs/2504.16580)<br/>
[Ignacio Peis](https://ipeis.github.io/),
[Batuhan Koyuncu](https://batukoyuncu.com/),
[Isabel Valera](https://ivaleram.github.io/)\,
[Jes Frellsen](https://frellsen.org/)



<p align="center">
<img src=assets/hd_decoder.png />
</p>


## ⚙️ Requirements


The easiest way to use our code is by creating a [conda](https://conda.io/) environment with our provided requirements file:

```
conda env create -f environment.yaml
conda activate ldm
```

If you experienced issues with the transformers or the torchmetric packages, we recommend you to force this pip installation after creating the env:
```
pip install torchmetrics==0.4.0 --no-deps
```

## 🧱 Data preparation

### CelebA-HQ
The CelebAHQ datasets can be downloaded from [here](https://www.kaggle.com/datasets/lamsimon/celebahq/data).

### ImageNet
We refer to the official [LDM repository](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#imagenet) to easily download and prepare ImageNet.

### 🪑 Chairs and 🌎 ERA5
The shapenet and ERA5 climate datasets can be downloaded at this [link](https://drive.google.com/drive/folders/1r_sk5auYvllSpDG9ZjroOG0SH0v5kPmM?usp=sharing). Credits to the authors of [GASP](https://arxiv.org/abs/2102.04776).


# 🚀 Training LDMI

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.

### Training I-VAE

Configs for training KL-regularized autoencoders for INRs are provided at `configs/ivae`.
Training can be started by running
```
python main.py --base configs/ivae/<config_spec>.yaml -t --gpus 0,    
```
We do not directly train VQ-regularized models. See the [taming-transformers](https://github.com/CompVis/taming-transformers) 
repository if you want to train your own VQGAN.

### Training LDMI 

In ``configs/ldmi/`` we provide configs for training LDMI on all datasets.
Training can be started by running

```shell script
python main.py --base configs/ldmi/<config_spec>.yaml -t --gpus 0,
``` 

### Hyper-Transforming
If you choose one of `configs/ldmi/imagenet_ldmi.yaml` or `configs/ldmi/celebahq256_ldmi.yaml`, a pre-trained [LDM](https://github.com/CompVis/latent-diffusion) model (VQ-F4 variant) will be loaded and the HD decoder will be trained according to our hyper-transforming method.  You can download the pretrained LDMs on [CelebA-HQ (256 x 256)](https://ommer-lab.com/files/latent-diffusion/celeba.zip) and [ImageNet](https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt) in the provided links.

## Running experiments

Experiments can by easily run by calling adding scripts to the `experiments` folder and calling them in the `run_experiment.py` script. An example for sampling Celeba-HQ images can be:
```shell script
python run_experiment.py --experiment experiments/configs/log_images.yaml --model_cfg configs/ldmi/celebahq256_ldmi.yaml --ckpt <my_checkpoint>.ckpt
``` 

## Comments 

- This project is largely based on the [latent-diffusion codebase](https://github.com/CompVis/latent-diffusion). We’re grateful to all its contributors for making it open source!

## Globe plots

By default, climate data samples are displayed using flat map projections. To render these samples on a globe, you can make use of the functions provided in `utils/viz/plots_globe.py`. This functionality depends on the [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/) library, which must be installed separately. For setup instructions, refer to the [official installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html).

## Visualizing 3D Samples

To run experiments involving 3D model rendering, make sure to install both [`mcubes`](https://github.com/pmneila/PyMCubes) (for marching cubes extraction) and [`pytorch3d`](https://github.com/facebookresearch/pytorch3d). Note that installing PyTorch3D may require extra steps depending on your PyTorch version—it’s not always available via `pip`. Refer to their [installation guide](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) for detailed instructions.

## BibTeX

```
@InProceedings{peis2025hyper,
  title={Hyper-Transforming Latent Diffusion Models},
  author={Peis, Ignacio and Koyuncu, Batuhan and Valera, Isabel and Frellsen, Jes},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  publisher={PMLR},
  year={2025}
}
```