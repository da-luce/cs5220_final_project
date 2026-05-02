#pragma once

#include "networks/model.h"
#include <torch/torch.h>
#include <iostream>

#ifdef USE_NCCL
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#endif

// RAII wrapper for MPI + NCCL lifetime. Exposes rank, world_size, and device().
// In CPU-only builds (no USE_NCCL) this is a trivial stub: rank=0, world_size=1.
struct DistributedContext {
    int rank       = 0;
    int world_size = 1;

#ifdef USE_NCCL
    ncclComm_t comm{};

    DistributedContext(int& argc, char**& argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        cudaSetDevice(rank);

        ncclUniqueId nccl_id;
        if (rank == 0) ncclGetUniqueId(&nccl_id);
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&comm, world_size, nccl_id, rank);

        if (rank == 0)
            std::cout << "[NCCL+MPI enabled] MPI world_size=" << world_size << std::endl;
    }

    ~DistributedContext() {
        ncclCommDestroy(comm);
        MPI_Finalize();
    }

    torch::Device device() const { return torch::Device(torch::kCUDA, rank); }
#else
    DistributedContext(int& /*argc*/, char**& /*argv*/) {
        std::cout << "[CPU-only, no MPI/NCCL] Compiled without USE_NCCL" << std::endl;
    }

    ~DistributedContext() = default;

    torch::Device device() const { return torch::kCPU; }
#endif

    DistributedContext(const DistributedContext&)            = delete;
    DistributedContext& operator=(const DistributedContext&) = delete;
};

// Average model weights across all ranks. No-op in CPU-only builds.
inline void sync_weights(networks::StrategoNet& net, const DistributedContext& ctx) {
#ifdef USE_NCCL
    torch::NoGradGuard no_grad;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (auto& param : net->parameters()) {
        ncclAllReduce(param.data_ptr<float>(), param.data_ptr<float>(),
                      param.numel(), ncclFloat, ncclSum, ctx.comm, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    for (auto& param : net->parameters())
        param.div_(ctx.world_size);
#else
    (void)net;
    (void)ctx;
#endif
}
