ARG PYTORCH_VERSION="1.12.0"
ARG CUDA_VERSION="11.3"
ARG CUDNN_VERSION="8"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ENV PROJECT=/mos4d
RUN mkdir -p $PROJECT
RUN mkdir -p /mos4d/logs

ENV DATA=/mos4d/data
RUN mkdir -p $DATA

# Install MinkowskiEngine Dependencies
RUN apt-get update \
    && apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" \
    && cd MinkowskiEngine \
    && python setup.py install --force_cuda --blas=openblas

# Install project related dependencies
WORKDIR $PROJECT
COPY . $PROJECT
RUN python3 -m pip install --editable . \
    && rm -rf $PROJECT

RUN pip install tensorboard

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user \
    && chown -R user:user /mos4d