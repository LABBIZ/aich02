#!/bin/bash

SYNC_DIR="/vagrant"

apt-get update
apt-get -y upgrade
apt-get -y install wget
apt-get -y install --no-install-recommends ccache cmake curl g++ make unzip git
apt-get -y install python3 python3-dev python3-pip python3-setuptools python3-virtualenv python3-numpy python3-scipy python3-yaml python3-h5py python-six
apt-get -y install libhdf5-dev libarchive-dev
pip3 install --upgrade pip

pip3 install tensorflow
pip3 install keras

pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl 
pip3 install torchvision

pip3 install nnabla

curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
tar xvf protobuf-v3.1.0.tar.gz
cd protobuf-3.1.0
mv BUILD _BUILD
mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
make
sudo make install

cd ${SYNC_DIR}

git clone https://github.com/gabime/spdlog.git
git checkout v0.13.0
cp -a spdlog/include/spdlog /usr/include/spdlog

git clone https://github.com/sony/nnabla.git
cd nnabla
git checkout v0.9.4
mkdir build && cd build
cmake .. -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_API=OFF
make
make install

cd ${SYNC_DIR}/nnabla/examples/vision/mnist
python3 classification.py

cd ${SYNC_DIR}/nnabla/examples/cpp/mnist_runtime
python3 save_nnp_classification.py
make
