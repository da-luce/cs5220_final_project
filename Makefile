TORCH_CMAKE ?= /global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/torch/share/cmake
BUILD_DIR   ?= build
BUILD_TYPE  ?= Release

# On Perlmutter, load modules before building:
#   module swap PrgEnv-cray PrgEnv-gnu
#   module load gcc/12.2.0
#   module load cray-mpich
#   rm -rf build && make configure
CC     ?= gcc
CXX    ?= g++
MPICC  ?= mpicc
MPICXX ?= mpicxx

.PHONY: all configure train clean

all: train

configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_PREFIX_PATH=$(TORCH_CMAKE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DMPI_C_COMPILER=$(MPICC) \
		-DMPI_CXX_COMPILER=$(MPICXX)

train: configure
	cmake --build $(BUILD_DIR) --target train_stratego -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
