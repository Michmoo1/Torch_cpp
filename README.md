# Torch_cpp

wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

cmake -DCMAKE_PREFIX_PATH=..(your absolute path)/Torch_cpp/libtorch

cmake --build . --config Release