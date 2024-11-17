# Mixture of Experts Meets Prompt-Based Continual Learning

This repository is the official implementation of `Mixture of Experts Meets Prompt-Based Continual Learning` (NeurIPS 2024).

Exploiting the power of pre-trained models, prompt-based approaches stand out compared to other continual learning solutions in effectively preventing catastrophic forgetting, even with very few learnable parameters and without the need for a memory buffer. While existing prompt-based continual learning methods excel in leveraging prompts for state-of-the-art performance, they often lack a theoretical explanation for the effectiveness of prompting. This paper conducts a theoretical analysis to unravel how prompts bestow such advantages in continual learning, thus offering a new perspective on prompt design. We first show that the attention block of pre-trained models like Vision Transformers inherently encodes a mixture of experts architecture, characterized by linear experts and quadratic gating score functions. This realization drives us to provide a novel view on prefix tuning, reframing it as the addition of new task-specific experts, thereby inspiring the design of a novel gating mechanism termed Non-linear Residual Gates (NoRGa). Through the incorporation of non-linear activation and residual connection, NoRGa enhances continual learning performance while preserving parameter efficiency. The effectiveness of NoRGa is substantiated both theoretically and empirically across diverse benchmarks and pretraining paradigms.

## Requirements

- Python 3.10.5

```setup
pip install -r requirements.txt
```

## Experimental Setup

Our code has been tested on four datasets: CIFAR-100, ImageNet-R, 5-Datasets, and CUB-200:

### Dataset

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- 5-Datasets (including SVHN, MNIST, CIFAR10, NotMNIST, FashionMNIST)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

### Supervised and Self-supervised Checkpoints

We incorporated the following supervised and self-supervised checkpoints as backbones:
- Sup-21K VIT
- [iBOT-21K](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint.pth)
- [iBOT](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
- [MoCo v3](https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link)
- [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)  
  
Please download the self-supervised checkpoints and put them in the `/checkpoints/{checkpoint_name}` directory, excecpt Sup-21K. 

**NOTE**: For iBOT, please rename the checkpoint file to `ibot_vitbase16_pretrain.pth`.


## Usage

To reproduce the results mentioned in our paper, execute the training script in `/scripts/{dataset}_{backbone}_{method}.sh`. e.g.

**NoRGa**: If you want to train with Sup-21K backbone, run the following command:

- Split CIFAR-100:
```bash
bash scripts/cifar100_Sup21k_NoRGa.sh
```

- Split CUB-200:
```bash
bash scripts/cub_Sup21k_NoRGa.sh
```

- Split ImageNet-R:
```bash
bash scripts/imr_Sup21k_NoRGa.sh
```

- 5-datasets:
```bash
bash scripts/5datasets_Sup21k_NoRGa.sh
```

If you encounter any issues or have any questions, please let us know. 

## Acknowledgement
This repository is developed mainly based on the PyTorch implementation of [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt). Many thanks to its contributors!