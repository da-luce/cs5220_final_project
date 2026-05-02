BUILD_DIR  ?= build
BUILD_TYPE ?= Release

PERLMUTTER_TORCH_CMAKE := /global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/torch/share/cmake
PERLMUTTER_NCCL_LIB    := /global/common/software/nersc9/pytorch/2.8.0/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

.PHONY: configure configure_perlmutter build clean

# Generic configure — let cmake find compilers on its own.
# Override torch path if needed: make configure TORCH_CMAKE=<path>
configure:
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
		$(if $(TORCH_CMAKE),-DCMAKE_PREFIX_PATH=$(TORCH_CMAKE)) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# Perlmutter profile — only sets what cmake can't discover itself on NERSC.
# Requires modules loaded first (must be done in your shell, not from make):
#   module swap PrgEnv-cray PrgEnv-gnu
#   module load gcc/12.2.0
#   module load cray-mpich
configure_perlmutter:
	@echo "NOTE: make sure you have loaded the required modules before building."
	@echo "  module swap PrgEnv-cray PrgEnv-gnu"
	@echo "  module load gcc/12.2.0"
	@echo "  module load cray-mpich"
	mkdir -p $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) \
		-DCMAKE_PREFIX_PATH=$(PERLMUTTER_TORCH_CMAKE) \
		-DNCCL_LIB=$(PERLMUTTER_NCCL_LIB) \
		-DBUILD_ITT_STUB=ON \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build:
	cmake --build $(BUILD_DIR) --target train_stratego -j$(shell nproc)

clean:
	rm -rf $(BUILD_DIR)
