#!/bin/bash
echo "=====  ENV SETUP START ====="
CUR_DIR=$(pwd)

# Install Miniconda locally
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda
source $PWD/miniconda/bin/activate

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment
conda create --name mask2former python=3.10 -y
conda activate mask2former

# Install PyTorch + Detectron2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Install Mask2Former dependencies
cd Mask2Former
pip install -r requirements.txt

# Build ops
cd mask2former/modeling/pixel_decoder/ops
bash make.sh

# Extra Python packages
pip install opencv-python
pip install "Pillow<10.0.0"
pip install "scikit-image<0.21.0"
pip install medpy
pip install seaborn
pip install "setuptools==69.5.1"
# Set PYTHONPATH
export PYTHONPATH=$CUR_DIR/Mask2Former:$PYTHONPATH

# Return to original directory
cd $CUR_DIR


echo "===== ENV SETUP COMPLETE ====="