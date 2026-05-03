#pragma once

#include "networks/model.h"
#include <torch/torch.h>
#include <iostream>

#ifdef USE_NCCL
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
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

// Collective termination check: returns true only when every rank reports done.
// Used to coordinate exit from training loops that contain collectives — if one
// rank exits early, the others hang in the next ncclAllReduce/MPI_Bcast.
inline bool all_ranks_done(bool local_done, const DistributedContext& ctx) {
#ifdef USE_NCCL
    if (ctx.world_size <= 1) return local_done;
    int l = local_done ? 1 : 0, g = 0;
    MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return g != 0;
#else
    (void)ctx;
    return local_done;
#endif
}

// Average model weights across all ranks. No-op in CPU-only builds.
inline void sync_weights(networks::StrategoNet& net, const DistributedContext& ctx) {
#ifdef USE_NCCL
    if (ctx.world_size <= 1) return;

    // Test
    // c10::cuda::getCurrentCUDAStream().synchronize();

    torch::NoGradGuard no_grad;
    // Run NCCL on PyTorch's current CUDA stream so the caching allocator is aware
    // of the work and subsequent tensor ops are correctly ordered after it.
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    ncclGroupStart();
    for (auto& param : net->parameters()) {
        ncclAllReduce(param.data_ptr<float>(), param.data_ptr<float>(),
                      param.numel(), ncclFloat, ncclSum, ctx.comm, stream);
    }
    ncclGroupEnd();
    for (auto& param : net->parameters())
        param.div_(ctx.world_size);
#else
    (void)net;
    (void)ctx;
#endif
}
