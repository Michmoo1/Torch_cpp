# Torch_cpp

For diamonds:
Download diamonds dataset than add matplot
sudo apt-get install python3-matplotlib python3-numpy python3-dev

wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

mkdir build

cd build

In build folder clone repo
git clone https://github.com/lava/matplotlib-cpp.git

cmake -DCMAKE_PREFIX_PATH=..(your absolute path)/Torch_cpp/libtorch ..

make