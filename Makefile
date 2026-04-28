TORCH_CMAKE ?= /global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/torch/share/cmake
BUILD_DIR   ?= build
BUILD_TYPE  ?= Release

# On Perlmutter, load a GCC ≥ 9 module before building:
#   module load gcc/12.2.0
#   rm -rf build && make configure
CC  ?= gcc
CXX ?= g++

.PHONY: all configure train clean

all: train

configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_PREFIX_PATH=$(TORCH_CMAKE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX)

train: configure
	cmake --build $(BUILD_DIR) --target train_stratego -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
