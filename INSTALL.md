### Install Anaconda

Follow the instruction to install anaconda [here](https://www.anaconda.com/download).

Follow the instruction [here](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) to set up Mamba, a fast environment solver for conda.

```
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Note: technically, the mamba solver should behave the same as the default solver. However, there have been cases where dependencies
can not be properly set up with the default mamba solver. The following instructions have **only** been tested on mamba solver.

### Install Gaussian-related packages

```
conda create -y -n gaussian_splatting python=3.8 && conda activate gaussian_splatting
conda install -c conda-forge gcc=10.3.0 --strict-channel-priority
conda install -c conda-forge cxx-compiler --strict-channel-priority
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 cudatoolkit=11.7 cudatoolkit-dev=11.7 -c pytorch -c conda-forge -c nvidia
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..
cd submodules/simple-knn
pip install .
cd ../..
pip install plyfile tqdm
```
