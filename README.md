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
1. Open the DeepSCan web interface using provided [link](https://deepscan.sidichenlab.org/)
2. Paste protein sequences in FASTA format into the input box, using examples below or sequences in demo dataset
    ```bash
    >ECD_P69354
    KGLSSTSIVYILIAVCLGGLIGIPALIC
    >hTMC_EPCAM_v3_10
    GLKAGVIAVLGIGIIAVVAGITFAVYSRKKRMAKYEKAEIKEDYQPYFRNHL
    >hTMC_EPCAM_v3_4
    GLKAGVIAVPVLLAIAVVAGILGYKRSRKKRMAKYEKAEIKENEAVMEVKAH
    >hTMC_Q9UHN6
    MYATDSRGHSPAFLQPQNGNSRHPSGYVPGKVVPLRPPPPPKSQASAKFTSIRREDRATFAFSPEEQQAQRESQKQKRHKNTFICFAITSFSFFIALAIILGISSKYAPDENCPDQNPRLR
    ```
4. Click the Run Prediction button
5. Once prediction is complete, click Download CSV button
6. Interpret the results. The output CSV file typically includes:
* **Sequence ID**: taken from the FASTA header
* **Sequence length**
* **Prediction scores**:
  * Higher score indicates stronger surface expression level
  * Score of 1 represents expression levels of HLA (Type I) or CLEC4F (Type II) reference modules. Scores are log-transformed
  * Display dual model scores from Genesis and Omni models
* **Genesis CST Category**: CST0–5 represent derived Genesis prediction score categories, defined using cutoffs at 0.3, 0.55, 0.8, 1.0, and 1.2
* **Topology**: describes how transmembrane protein is arranged on cell surface.
  * Type I topology: Extracellular domain is at N terminus
  * Type II topology: Extracellular domain is at C terminus
* **TMD sequence**: only available in CSV file. Shows transmembrane domain (TMD) of single-pass transmembrane protein (SPTM)

![DeepSCan_webserver](docs/_static/img/DeepSCan_webserver_v3.png)



![235_CSD](docs/_static/img/235_CSD_v2.png)

---

## DeepSCan Docker image installation
1. Pulling DeepSCan docker image from the docker hub repository
    ```bash
    docker pull zf77/deepscan:stable
    ```
2. Start a new Docker container and run Genesis model prediction using 
    ```bash
    docker run --rm --gpus all -it zf77/deepscan:stable micromamba run -n torch python Genesis_Quant_8k_model.py --input Quant_8k_training_set_first800.csv --out Quant_8k_training_set_first800_predicted.csv
    ```
3. At this point you are in the docker environment and can run the Genesis model prediction using your datasets in csv file. An example:
    ```bash
    micromamba run -n torch python Genesis_Quant_8k_model.py --input Quant_8k_training_set_first800.csv --out Quant_8k_training_set_first800_predicted.csv
    ```
    

### Operating systems
- Linux is the native operating system
- Docker image can be run using Docker on macOS or WSL on Windows

### Software dependencies
All required software packages are preinstalled in the docker image. This project assumes Python + PyTorch-based workflow. 

**Essential packages:**
- Python: 3.11
- PyTorch: 2.5.1
- PyTorch-CUDA: 12.4
- Common packages:
  - numpy
  - pandas
  - scipy
  - tqdm
  - fair-esm
  - conda-forge::transformers
  - anaconda::evaluate
  - anaconda::scikit-learn
  - jupyterlab
  - ipykernel

### Hardware requirements
**Standard (CPU)**
- Any modern desktop/laptop CPU
- RAM: 8–16 GB recommended

**Accelerated (GPU, recommended for speed)**
- NVIDIA GPU with **>= 12 GB VRAM** recommended
- Non-standard hardware: none

---

## 📃 Citation
- [AI-guided de novo design and generation of potent cell surface display elements (to be published)]
```bibtex
@article{Fang2026deepscan,
title = {AI-guided de novo design and generation of potent cell surface display elements},
author = {Zhenhao Fang and Joshua Saskin and Seok-Hoon Lee and Charles Zou and Shan Xin and Xiaoyu Huang and Xingxin Pan and Chuanpeng Dong and Ardavan Abiri and Nidhi Sahni and Yanzhi Feng and Lei Peng and S. Stephen Yi and Sidi Chen},
year = {2026},
}
```
