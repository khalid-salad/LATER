# LATER
Linear Algebra on TEnsoRcore;
see http://www2.cs.uh.edu/~panruowu/later.html

## Prerequisites

* CMake 3.12+
* CUDA 11.1+

## Build
### Linux
#### Clone Repo
```console
$ git clone git@github.com:Orgline/LATER.git
```
#### Create Build Directory
```console
$ cd LATER
$ mkdir -p build && cd build
```
#### Set Environment Variables
```console
$ export CUDA_PATH=/usr/local/cuda
```
Change the CUDACXX and CUDA_PATH environment variables to match
your system's CUDA installation directory. Set the CUTLASS_DIR environment
variable to match your system's CUTLASS installation directory. 

#### Build
```console
$ cmake .. -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_80,code=sm_80" -DCUDA_ARCH="Ampere"
$ # On Volta, 75->70, Turing->Volta
$ cmake --build .
```

### Windows 

```console
$ git clone git@github.com:Orgline/LATER.git
$ cd LATER && mkdir build && cd build
$ cmake .. -A x64
$ cmake --build .
```
## Run tests
### Linux
```console
$ cd build/test
$ ./test_qr 1 16384 16384 -check
$ ./test_potrf 16384 -check
```

### Windows
```console
$ cd test/debug
$ test_qr.exe 1 16384 16384 -check
$ test_potrf.exe 16384 -check
```

## Tested GPUs and Platforms
* V100 (on RHEL Linux 7, CUDA 10.1, GCC 8)
* Titan V (on Ubuntu 18.04 Linux, CUDA 10.1, GCC 7.5.0)
* GeForce RTX 2060 (on Windows 10, CUDA 10.2, Visual Studio 2017)
* GeForce RTX 2080 Super (Ubuntu 18.04 Linux, CUDA 10.2, GCC 7.5.0)
