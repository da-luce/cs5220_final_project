#pragma once

// Lightweight scope-timer profiler for the training loop.
//
// Each PROFILE_SCOPE / PROFILE_SCOPE_GPU constructs a ScopedTimer that
// accumulates wall-clock time into a named bucket in the global Profiler
// singleton. PROFILE_SCOPE_GPU additionally issues cudaDeviceSynchronize()
// at scope entry and exit so the measured interval reflects the full
// H2D/kernel/D2H duration rather than just the async launch latency.
//
// The profiler is process-local — each rank has its own counters. Rank 0
// is responsible for serializing the final summary into the metrics log.

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iomanip>

#ifdef USE_NCCL
#include <cuda_runtime.h>
#endif

namespace profiling {

struct StageStat {
    double total_ms = 0.0;
    int64_t count   = 0;
};

class Profiler {
public:
    static Profiler& instance() {
        static Profiler p;
        return p;
    }

    void record(const std::string& name, double ms) {
        auto it = stats_.find(name);
        if (it == stats_.end()) {
            order_.push_back(name);
            stats_.emplace(name, StageStat{ms, 1});
        } else {
            it->second.total_ms += ms;
            it->second.count += 1;
        }
    }

    const std::vector<std::string>& order() const { return order_; }
    const std::unordered_map<std::string, StageStat>& stats() const { return stats_; }

    // Emit a single JSONL record containing per-stage totals and counts.
    // Designed to be appended to the existing metrics log so it can be
    // parsed alongside the per-batch entries.
    std::string to_jsonl(double total_time_ms) const {
        std::ostringstream o;
        o.setf(std::ios::fixed);
        o << std::setprecision(3);
        o << "{\"profiling\":true";
        o << ",\"total_time_ms\":" << total_time_ms;
        o << ",\"stages\":{";
        bool first = true;
        for (const auto& name : order_) {
            const auto& s = stats_.at(name);
            if (!first) o << ",";
            first = false;
            o << "\"" << name << "\":{\"total_ms\":" << s.total_ms
              << ",\"count\":" << s.count << "}";
        }
        o << "}}\n";
        return o.str();
    }

private:
    std::unordered_map<std::string, StageStat> stats_;
    std::vector<std::string> order_;
};

class ScopedTimer {
public:
    ScopedTimer(const char* name, bool sync_cuda)
        : name_(name), sync_(sync_cuda) {
#ifdef USE_NCCL
        if (sync_) cudaDeviceSynchronize();
#else
        (void)sync_;
#endif
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
#ifdef USE_NCCL
        if (sync_) cudaDeviceSynchronize();
#endif
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        Profiler::instance().record(name_, ms);
    }

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    bool sync_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace profiling

#define PROFILER_CONCAT_INNER(a, b) a##b
#define PROFILER_CONCAT(a, b) PROFILER_CONCAT_INNER(a, b)

#define PROFILE_SCOPE(name) \
    ::profiling::ScopedTimer PROFILER_CONCAT(_prof_, __LINE__){name, false}

#define PROFILE_SCOPE_GPU(name) \
    ::profiling::ScopedTimer PROFILER_CONCAT(_prof_, __LINE__){name, true}
