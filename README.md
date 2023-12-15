<!--
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# pyTorch wrapper for NVIDIA nvBlox

This package connects [nvblox](https://github.com/nvidia-isaac/nvblox)'s mapping and collision query functions with pytorch. This package 
supports both mapping a world from a depth camera and also querying the built world for 
signed distances. Read [nvblox](https://github.com/nvidia-isaac/nvblox) for more information on how 
it works.

Checkout [CuRobo](https://curobo.org) for examples of integrating this package with motion generation
for manipulators.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## Updates
 - December 2023: Updated wrappers to work with nvblox 0.0.5 and added instructions to compile with PRE_CXX11_ABI.

## Citation

If you found this work useful, please cite the below report,

```
@article{curobo_report23,
         title={CuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation},
         author={Sundaralingam, Balakumar and Hari, Siva Kumar Sastry and 
         Fishman, Adam and Garrett, Caelan and Van Wyk, Karl and Blukis, Valts and 
         Millane, Alexander and Oleynikova, Helen and Handa, Ankur and 
         Ramos, Fabio and Ratliff, Nathan and Fox, Dieter},
         journal={arXiv preprint},
         year={2023}
        }
```


## Code Contributors

- Valts Blukis
- Balakumar Sundaralingam
- Alexander Millane

## Docker
We have found docker to be the most stable way to use nvblox_torch. Docker instructions are 
at [docker_development](https://curobo.org/source/getting_started/5_docker_development.html). The dockerfile is in [curobo_github](https://github.com/NVlabs/curobo/blob/main/docker/x86.dockerfile). 
There are instructions in the link to use nvblox_torch on NVIDIA Jetson and also with 
NVIDIA Isaac Sim.

## Install Instructions

pyTorch that is available through pip wheels and also with Isaac Sim has been compiled with `D_GLIBCXX_USE_CXX11_ABI=0`. 
pyTorch that's available through docker containers at [ngc](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) are compiled with `D_GLIBCXX_USE_CXX11_ABI=1`. You can check what value was used for your pytorch installation with 
`python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`. 

### Prequisites:

If you are on Ubuntu older than 20.04:

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```

### Installation for CXX11_ABI

1. Install dependenices

    ```
    sudo apt-get install libgoogle-glog-dev libgtest-dev libsqlite3-dev curl tcl libbenchmark-dev
    ```
2. Install nvblox

    ```
    git clone https://github.com/valtsblukis/nvblox.git && cd nvblox/nvblox && mkdir build && \
    cmake .. \
    -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
    && make -j32 && \
    sudo make install
    ```

3. Install this repository

    ```
    git clone https://github.com/NVlabs/nvblox_torch.git && cd nvblox_torch
    sh install.sh $(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')
    python -m pip install -e .
    ```

### Installation for PRECXX11_ABI

The below instructions can also be used to install nvblox torch with Isaac Sim. Change all instances of `python` to `omni_python`, 
where `omni_python` maps to the python shell of your Isaac Sim installation as `alias omni_python='~/.local/share/ov/pkg/isaac_sim-2023.1.0/python.sh'`.


1. Create environment variable that stores the value of `CXX11_ABI` of pytorch installation:

    ```
    export TORCH_CXX11=0 # change this value (0=False, 1=True) based on python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
    ```

2. Create environment variables that will store the path you want to install nvblox and 
also the value of CXX11_ABI:

    ```
    export PKGS_PATH=/home/${USER}/pkgs
    mkdir -p ${PKGS_PATH}
    ```


3. Update cmake with:
    ```
    cd ${PKGS_PATH} && wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz && \
        tar -xvzf cmake-3.27.1.tar.gz && \
        sudo apt update &&  sudo apt install -y build-essential checkinstall zlib1g-dev libssl-dev && \
        cd cmake-3.27.1 && ./bootstrap && \
        make -j8 && \
        sudo make install
    ``` 

4. Install sqlite3:

    ```
    cd ${PKGS_PATH} && git clone https://github.com/sqlite/sqlite.git -b version-3.39.4 && \
        cd ${PKGS_PATH}/sqlite && CFLAGS=-fPIC ./configure --prefix=${PKGS_PATH}/sqlite/install/ && \
        make -j8 && make install
    ```

5. Install glog:
    ```
    cd ${PKGS_PATH} && git clone https://github.com/google/glog.git -b v0.6.0 && \
    cd glog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=${PKGS_PATH}/glog/install/ \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11} \
    && make -j8 && make install
    ```

6. Install gflags:
    ```
    cd ${PKGS_PATH} && git clone https://github.com/gflags/gflags.git -b v2.2.2 && \
    cd gflags &&  \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=${PKGS_PATH}/gflags/install/ \
    -DGFLAGS_BUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11} \
    && make -j8 && make install
    ```

7. Install nvblox:

    ```
    cd ${PKGS_PATH} &&  git clone https://github.com/valtsblukis/nvblox.git && cd ${PKGS_PATH}/nvblox/nvblox mkdir build && cd build && \
    cmake ..  -DBUILD_REDISTRIBUTABLE=ON \
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0  -DPRE_CXX11_ABI_LINKABLE=ON \
    -DSQLITE3_BASE_PATH="${PKGS_PATH}/sqlite/install/" -DGLOG_BASE_PATH="${PKGS_PATH}/glog/install/" \
    -DGFLAGS_BASE_PATH="${PKGS_PATH}/gflags/install/" -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 && \
    make -j32 && \
    sudo make install
    ```


8. Install nvblox_torch in your python environment:

    ```
    cd ${PKGS_PATH} &&  git clone https://github.com/NVlabs/nvblox_torch.git && cd nvblox_torch
    sh install.sh $(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')
    python -m pip install -e .
    ```


## Examples
To reduce debug printing, use `export GLOG_minloglevel=2`

**1. SDF from dummy map**

``` python examples/get_sdf.py```


For the remaining demos, you need to download the Sun3D dataset:

```
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P datasets/3dmatch
unzip datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d datasets/3dmatch
```


**2. Sun3D data**
Segmenting sofa as a dynamic class

```
python examples/run_mapper.py \
  --dataset "datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2" \
  --dataset-format "sun3d" \
  --voxel-size 0.04 \
  --decay-occupancy-every-n -1 \
  --dynamic-class "sofa" \
  --visualize-voxels
```


**3. Realsense**
With human segmentation

```
python examples/run_mapper.py \
    --dataset "sun3d-mit_76_studyroom-76-1studyroom2" \
    --dataset-format realsense \
    --voxel-size 0.04 \
    --decay-occupancy-every-n 1 \
    --dynamic-class person \
    --visualize-voxels
```

**4. Mesh Input**
Using PyRender to generate depth images. No segmentation.You need to provide your own mesh file here.

```
python examples/run_mapper.py \
  --dataset "mesh.stl" \
  --dataset-format mesh \
  --voxel-size 0.01 \
  --decay-occupancy-every-n -1 \
  --integrator-type occupancy \
  --visualize-voxels \
  --clear-map-every-n -1
```

