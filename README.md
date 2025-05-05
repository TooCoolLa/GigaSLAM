# GigaSLAM

This repository contains the source code for our papers:

[GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats](https://arxiv.org/abs/2503.08071)

##  Setup and Installation

### Hardware and System Environment

This project was developed, tested, and run in the following hardware/system environment

```
Hardware Environment：
	CPU(s)： Intel Xeon(R) Gold 6128 CPU @ 3.40GHz × 12
	GPU(s)： NVIDIA RTX 4090 (24 GiB VRAM) / NVIDIA L20 (48 GiB GDDR6)
	RAM：67.0 GiB (DDR4, 2666 MT/s) / 128.0 GiB (DDR4, 3200 MT/s)
	Disk: Dell 8TB 7200RPM HDD (SATA, Seq. Read 220 MiB/s)

System Environment：
	Linux System: Ubuntu 22.04.3 LTS
	CUDA Version: 11.8
	cuDNN Version: 9.1.0
	NVIDIA Drivers: 555.42.06
	Conda version: 23.9.0 (Miniconda)

Compilers & Build Tools:
    NVIDIA CUDA Compiler (nvcc): V11.8.89
    C++ Compiler: GCC/G++ 11.4.0
    GNU Make Version： 4.3
    Cmake Version 3.22.1
```

As part of the project code relies on `CUDA/C++`, please ensure your compilation environment is properly workable.

### Environment Setup

#### Step 1: Dependency Installation

Creating a virtual environment using conda (or miniconda)

```bash
conda create -n gigaslam python=3.10
conda activate gigaslam
# pip version created by conda: 25.1
```

Next, install `PyTorch`

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# Verified to work with CUDA 11.8 and torch 2.2.0
```

Then install other dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter installation issues with `torch_scatter` (which may happen in certain network environments), comment out the corresponding line in `requirements.txt` and manually download & install the wheel from link: https://pytorch-geometric.com/whl/

```bash
wget https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_scatter-2.1.2%2Bpt22cu118-cp310-cp310-linux_x86_64.whl  
# Alternatively, you could manually download via your browser
pip install ./torch_scatter-2.1.2+pt22cu118-cp310-cp310-linux_x86_64.whl
```


> **Special Note Regarding Version Compatibility:**
> Particular attention must be paid to the `xformers-0.0.24` dependency specified in `requirements.txt`. If you attempt to install alternative versions of `torch` (other than the explicitly recommended `2.2.0`), subsequent installations of `xformers` via `pip` will **force-uninstall** your current `PyTorch` version and replace it with the specific version deemed compatible by `xformers`. This forced version downgrade/upgrade poses critical risks to `CUDA/C++` compilation workflows, as PyPI-distributed `xformers` packages are **pre-compiled binaries** tightly coupled to specific CUDA toolkit and `PyTorch` versions.
>
> **Extended Compatibility Guidance:**
> For developers requiring alternative `xformers` or `PyTorch` versions, consult the [xFormers GitHub Repository](https://github.com/facebookresearch/xformers) for version relationships. While the repository lacks an official compatibility reference, we have curated version compatibility references by cross-referencing release notes. The following table may assist both GigaSLAM users and general developers who are encountering similar dependency conflicts:
>
> | `xFormer` | `PyTorch` | `xFormer` | `PyTorch` |
> | :-------: | :-------: | :-------: | :-------: |
> |  0.0.21   |   2.0.1   |  0.0.26   |   2.2.0   |
> |  0.0.22   |   2.0.1   |  0.0.27   |   2.3.0   |
> |  0.0.23   |   2.1.1   |  0.0.28   |   2.4.1   |
> |  0.0.24   |   2.2.0   |  0.0.29   |   2.5.1   |
> |  0.0.25   |   2.2.0   |  0.0.30   |   2.7.0   |

#### Step 2: Compiling the `CUDA/C++` Modules

Proceed to install the required `CUDA/C++` components. **Thoroughly verify your build environment** – incompatible compiler versions, conflicting dependencies, or mismatched CUDA toolkits may cause compilation failures or unexpected binary behavior. 

##### Step 2.1: Compile 3D GS Rendering Module

```
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

##### Step 2.2: Compile Loop-Closure Detection Module

1. Install the OpenCV C++ API.

```
sudo apt-get install -y libopencv-dev
```

2. Install DBoW2

```
cd DBoW2
mkdir -p build && cd build
cmake ..
make
sudo make install
cd ../..
```

3. Install the image retrieval

```
pip install ./DPRetrieval
```

##### Step 2.3: Compile Loop-Closure Correction Module

```bash
python setup.py install
```

> **Important Notes:** This project is implemented based on the MonoGS framework, whose native architecture primarily targets small-scale scenes. In terms of hardware requirements, processing ultra-long sequence data significantly increases CPU RAM load - through targeted optimizations, we have stabilized the CPU memory consumption of 4000-frame KITTI datasets at approximately 10 GiB. However, longer sequences may still require additional memory resources. We strongly recommend running this system on server environments equipped with 32+ GiB CPU RAM. For **personal computers**  (particularly common 16 GiB CPU RAM setups), please continuously monitor memory usage via System Monitor (or similar) to prevent sudden memory spikes from affecting other system processes.
>
> We are currently exploring methods to further optimize memory consumption to achieve better operational efficiency.

#### Step 3: Running

Run the code by using the following command, and the pre-training weights of models such as `DISK`, `LightGlue` and `UniDepth` will be downloaded during the first run.

```bash
python slam.py --config ./path_to_your_config.yaml
```

## Acknowledgements

Our project is based on [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [MonoGS](https://github.com/muskie82/MonoGS), [DF-VO](https://github.com/Huangying-Zhan/DF-VO), [DPVO (DPV-SLAM)](https://github.com/princeton-vl/DPVO). Our work would not have been possible without these excellent repositories.

## Citation

```bibtex
@article{deng2025gigaslam,
  title={GigaSLAM: Large-Scale Monocular SLAM with Hierachical Gaussian Splats},
  author={Deng, Kai and Yang, Jian and Wang, Shenlong and Xie, Jin},
  journal={arXiv preprint arXiv:2503.08071},
  year={2025}
}
```
