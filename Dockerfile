FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Config
ENV ROS_DISTRO noetic

# Minimal setup
RUN apt update --fix-missing && apt install -y locales lsb-release && apt clean

ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

# Setup torch, cuda for the model and other dependencies
RUN apt install -y python3-pip && \
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install ROS
# [ROS] a. Setup your sources.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# [ROS] b. Set up the keys
RUN apt install -y curl && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# [ROS] c. Installation
RUN apt update && apt install -y --no-install-recommends ros-${ROS_DISTRO}-ros-base \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Install catkin tools and other packages 
RUN apt update && \
    apt install -y ros-${ROS_DISTRO}-catkin python3-catkin-tools ros-${ROS_DISTRO}-ros-numpy

# install ros packages
RUN apt update && apt install -y --no-install-recommends nano build-essential \
    libomp-dev libboost-all-dev ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-tf2 ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-geometry-msgs  \
    ros-${ROS_DISTRO}-eigen-conversions ros-${ROS_DISTRO}-tf-conversions python3 python3-venv \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

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

# Update the package repository and install dependencies
RUN apt update && apt install -y --no-install-recommends \
    git ninja-build cmake libopenblas-dev xauth openssh-server \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" \
    && cd MinkowskiEngine \
    && python3 setup.py install --force_cuda --blas=openblas

# Install project related dependencies
WORKDIR $PROJECT
COPY . $PROJECT
RUN python3 setup.py develop \
    && rm -rf $PROJECT 

RUN pip install tensorboard 

# Set numpy version to 1.20.1 as higher version cause issues in ros-numpy package 
RUN pip3 install --upgrade numpy==1.20.1    

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid 1000 user \
    && adduser --disabled-password --gecos '' --uid 1000 --gid 1000 user \
    && chown -R user:user /mos4d