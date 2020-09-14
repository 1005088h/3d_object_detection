## Install

### 1. Clone code

```bash
https://github.com/1005088h/object3d_det.git
cd ./object3d_det/
```

### 2. Install dependence python packages

It is recommend to use Anaconda package manager.

```bash
conda install scikit-image scipy numba pillow matplotlib
```

```bash
pip install opencv-python
```

### 3. Setup cuda for numba (will be removed in 1.6.0 release)

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```
