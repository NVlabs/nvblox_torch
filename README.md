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
nvblox works.

Checkout [CuRobo](https://curobo.org) for examples of integrating this package with motion generation
for manipulators.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

Use [Discussions](https://github.com/NVlabs/nvblox_torch/discussions) for questions on using this package.

Use [Issues](https://github.com/NVlabs/nvblox_torch/issues) if you find a bug.

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

## Install Instructions

1. Install nvblox with the ABI-compatibility mode

```
git clone https://github.com/valtsblukis/nvblox.git 
sudo apt-get install -y libgoogle-glog-dev libgtest-dev libgflags-dev python3-dev libsqlite3-dev
cd /usr/src/googletest && sudo cmake . && sudo cmake --build . --target install
cd nvblox && cd nvblox && mkdir build && cd build
cmake .. -DPRE_CXX11_ABI_LINKABLE=ON
make -j32
sudo make install
```

If you get a `thrust` error in esdf_integrator.cu during compilation, you need to recompile nvblox with ABI compatibility.

If you get a C++ standard error, change `CMAKE_CXX_STANDARD` in line 5 of `nvblox_torch/cpp/CMakeLists.txt` to the 
specific standard from the error. 

2. Install nvblox_torch in your python environment:

```
sh install.sh
```


## Install with Isaac Sim:

### Upgrade cmake

  1. `wget https://cmake.org/files/v3.19/cmake-3.19.5.tar.gz`
  2. `tar -xvzf cmake-3.19.5.tar.gz`
  3. `sudo apt install build-essential checkinstall zlib1g-dev libssl-dev`
  4. `cd cmake-3.19.5 && ./bootstrap`
  5. `make -j8`
  6. `sudo make install`



### Install nvblox
Install nvblox with:

1. `cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_70,code=sm_70" -DCMAKE_CUDA_ARCHITECTURES="70" -DSTDGPU_CUDA_ARCHITECTURE_FLAGS_USER="70" -DBUILD_FOR_ALL_ARCHS=ON`

### Install nvblox_torch

`export TORCH_CUDA_ARCH_LIST=7.0+PTX`

`sh install_isaac_sim.sh $(omni_python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')`

`omni_python -m pip install -e .`

If you get a glog library not found runtime error:

`export LD_LIBRARY_PATH=/usr/local/lib`

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

