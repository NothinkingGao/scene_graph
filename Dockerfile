ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm

# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

WORKDIR /maskrcnn-benchmark

# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

RUN conda create --name scene_graph_benchmark \
    && conda activate scene_graph_benchmark \
    && conda install ipython \
    && conda install scipy \
    && conda install h5py \
    && pip install ninja yacs cython matplotlib tqdm opencv-python overrides \
    && conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch \

    && export INSTALL_DIR=$PWD \
    && cd $INSTALL_DIR \
    && git clone https://github.com/cocodataset/cocoapi.git \
    && cd cocoapi/PythonAPI \
    && python setup.py build_ext install \

    && cd $INSTALL_DIR \
    && git clone https://github.com/NVIDIA/apex.git \
    && cd apex \
    && git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac \
    && python setup.py install --cuda_ext --cpp_ext \
    && cd $INSTALL_DIR \
    && git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git \
    && cd scene-graph-benchmark \
    && python setup.py build develop \
    && unset INSTALL_DIR
