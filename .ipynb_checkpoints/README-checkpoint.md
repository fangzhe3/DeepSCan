# DeepSCan platform

## 📰 News
| 🗞️ News        | 📝 Description                 |
| --            | ------      |
| Hugging face release  | We release the Genesis Quant 8k attention pooling model on [Hugging Face](https://huggingface.co/zfan3/Genesis_Quant_8k_attention_pooling) |
| Docker image release  | We launch the first DeepSCan stable image in [Docker](https://hub.docker.com/r/zf77/deepscan/tags) |
| First Github release | **DeepSCan platform** is released on [Github](https://github.com/fangzhe3/DeepSCan) |

## Introduction
We are excited to announce the release of **DeepSCan platform**, a powerful tool that predicts cell surface translocation properties of single-pass transmembrane proteins (SPTM) from amino acid sequences. This repository contains the model training, inference and computational analysis codes, plus example demo.

DeepSCan is now available on [GitHub](https://github.com/fangzhe3/DeepSCan), and we welcome your star🌟!

We have prepared three demo scenarios for you:
| Scenario | Link | Processing speed | Testing environment
| --                      | ------    | ------    | ------    |
| DeepSCan webserver | [Link](https://deepscan.sidichenlab.org/) | 20 sequences, 1000 tokens, 8 seconds | Genesis & Omni model on AWS EC2 t3a CPU 
| DeepSCan docker image (CPU) | [Link](https://hub.docker.com/r/zf77/deepscan/tags) | 800 sequences, 139263 tokens, 138 seconds | Genesis model on Intel i7 13700
| DeepSCan docker image (GPU) | [Link](https://hub.docker.com/r/zf77/deepscan/tags) | 800 sequences, 139263 tokens, 12 seconds | Genesis model on GeForce RTX 4050

**Notes**: DeepSCan docker image supports both CPU & GPU excution (when a compatible GPU is available). Processing speed depends on the runtime environment. The benchmark specifications shown above were measured on a Dell XPS 15 9530.

---

# ⚡ Quick start
## DeepSCan webserver
1. Pulling a docker image from a docker hub repository
    ```bash
    docker pull zf77/deepscan:stable
    ```
2. Start a new Docker container
    ```bash
    docker run -it --name <container name> -v <Mounted local directory>:/app pyqlib/qlib_image_stable:stable
    ```

```
![DeepSCan_webserver](docs/_static/img/DeepSCan_webserver_v2.png)
***

  
## DeepSCan Docker images
1. Pulling a docker image from a docker hub repository
    ```bash
    docker pull zf77/deepscan:stable
    ```
2. Start a new Docker container
    ```bash
    docker run -it --name <container name> -v <Mounted local directory>:/app pyqlib/qlib_image_stable:stable
    ```
3. At this point you are in the docker environment and can run the qlib scripts. An example:
    ```bash
    >>> python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
    >>> python qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    ```
    
### Operating systems
- Linux (recommended)
- macOS (CPU inference supported)
- Windows (CPU inference supported; GPU support depends on CUDA/PyTorch build)

### Software dependencies
This project assumes Python + PyTorch-based workflow.

**Minimum**
- Python: **3.10+**
- PyTorch: **2.1+**
- CUDA: **11.8+** (only if using NVIDIA GPU)
- Common packages:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - tqdm
  - pyyaml
  - biopython (FASTA handling)

**If using ESM / transformer encoders**
- fair-esm or esm library (version pinned in `environment.yml` / `requirements.txt`)
- transformers (optional; only if used)

> If you provide `environment.yml` and/or `requirements.txt`, pin exact versions there.

### Tested versions
Fill in what you have actually tested:
- OS: Ubuntu 22.04 / 20.04
- Python: 3.10.x
- PyTorch: 2.1.x / 2.2.x
- CUDA: 11.8
- NVIDIA driver: 535+ (if GPU)

### Hardware requirements
**Standard (CPU)**
- Any modern desktop/laptop CPU
- RAM: 8–16 GB recommended

**Accelerated (GPU, recommended for speed)**
- NVIDIA GPU with **>= 12 GB VRAM** recommended (works with less depending on model/batch size)
- Non-standard hardware (if applicable): none

---

## 2. Installation guide

### Option A (recommended): conda environment
1. Install Miniconda/Anaconda.
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate deepscan


## 📃 Citation
- [AI-guided de novo design and generation of potent cell surface display elements (to be published)]
```bibtex
@article{Fang2026deepscan,
title = {AI-guided de novo design and generation of potent cell surface display elements},
author = {Zhenhao Fang and Joshua Saskin and Seok-Hoon Lee and Charles Zou and Shan Xin and Xiaoyu Huang and Xingxin Pan and Chuanpeng Dong and Ardavan Abiri and Nidhi Sahni and Yanzhi Feng and Lei Peng and S. Stephen Yi and Sidi Chen},
year = {2026},
}
```
