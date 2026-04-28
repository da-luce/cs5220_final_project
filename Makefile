TORCH_CMAKE ?= /global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/torch/share/cmake
BUILD_DIR   ?= build
BUILD_TYPE  ?= Release

.PHONY: all configure train clean

all: train

configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_PREFIX_PATH=$(TORCH_CMAKE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

train: configure
	cmake --build $(BUILD_DIR) --target train_stratego -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
