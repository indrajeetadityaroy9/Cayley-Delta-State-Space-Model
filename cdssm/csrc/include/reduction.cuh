// CDSSM CUDA Reduction Utilities
// Warp-level and block-level reductions for efficient gradient accumulation
#pragma once

#include "common.cuh"

namespace cdssm {

// Block-Level Reductions

// Block-level sum reduction using shared memory
// Requires shared memory of size BLOCK_SIZE floats
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val, float* shared_mem) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");

    const int tid = threadIdx.x;

    // Store to shared memory
    shared_mem[tid] = val;
    __syncthreads();

    // Tree reduction
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Return result (only thread 0 has valid result)
    return shared_mem[0];
}

// Block-level max reduction
template<int BLOCK_SIZE>
__device__ float block_reduce_max(float val, float* shared_mem) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");

    const int tid = threadIdx.x;

    shared_mem[tid] = val;
    __syncthreads();

    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }

    return shared_mem[0];
}

// Two-Stage Reduction (for larger blocks)

// First stage: warp-level reduction
// Second stage: block-level reduction across warps
template<int BLOCK_SIZE>
__device__ float block_reduce_sum_two_stage(float val, float* warp_results) {
    static_assert(BLOCK_SIZE <= 1024, "BLOCK_SIZE must be <= 1024");

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Stage 1: Warp-level reduction
    float warp_sum = warp_reduce_sum(val);

    // Store warp results to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = warp_sum;
    }
    __syncthreads();

    // Stage 2: First warp reduces across warp results
    float block_sum = 0.0f;
    if (warp_id == 0) {
        float warp_val = (lane_id < num_warps) ? warp_results[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(warp_val);
    }

    // Broadcast result to all threads
    if (tid == 0) {
        warp_results[0] = block_sum;
    }
    __syncthreads();

    return warp_results[0];
}

// Vectorized Reductions (for 2-element state vectors)

// Reduce pairs of values (for 2D state gradients)
struct Float2 {
    float x, y;

    __device__ __forceinline__ Float2() : x(0.0f), y(0.0f) {}
    __device__ __forceinline__ Float2(float x_, float y_) : x(x_), y(y_) {}

    __device__ __forceinline__ Float2 operator+(const Float2& other) const {
        return Float2(x + other.x, y + other.y);
    }

    __device__ __forceinline__ Float2& operator+=(const Float2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
};

__device__ __forceinline__ Float2 warp_reduce_sum(Float2 val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

// Atomic Add with Pre-Reduction

// Reduce within warp before atomic add to global memory
// Reduces contention when multiple threads write to same location
__device__ __forceinline__ void atomic_add_warp_reduced(
    float* global_ptr,
    float val
) {
    // Warp-level reduction
    float reduced = warp_reduce_sum(val);

    // Only lane 0 does the atomic add
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        atomicAdd(global_ptr, reduced);
    }
}

// Block-level pre-reduction before atomic add
template<int BLOCK_SIZE>
__device__ __forceinline__ void atomic_add_block_reduced(
    float* global_ptr,
    float val,
    float* shared_mem
) {
    float reduced = block_reduce_sum<BLOCK_SIZE>(val, shared_mem);

    // Only thread 0 does the atomic add
    if (threadIdx.x == 0) {
        atomicAdd(global_ptr, reduced);
    }
}

// Welford's Algorithm for Online Mean/Variance

struct WelfordState {
    float mean;
    float M2;      // Sum of squared differences from mean
    int count;

    __device__ __forceinline__ WelfordState() : mean(0.0f), M2(0.0f), count(0) {}
};

// Update Welford state with new sample
__device__ __forceinline__ void welford_update(
    WelfordState& state,
    float sample
) {
    state.count++;
    float delta = sample - state.mean;
    state.mean += delta / state.count;
    float delta2 = sample - state.mean;
    state.M2 += delta * delta2;
}

// Combine two Welford states (for parallel reduction)
__device__ __forceinline__ WelfordState welford_combine(
    const WelfordState& a,
    const WelfordState& b
) {
    WelfordState result;
    result.count = a.count + b.count;

    if (result.count == 0) {
        result.mean = 0.0f;
        result.M2 = 0.0f;
        return result;
    }

    float delta = b.mean - a.mean;
    result.mean = (a.count * a.mean + b.count * b.mean) / result.count;
    result.M2 = a.M2 + b.M2 + delta * delta * a.count * b.count / result.count;

    return result;
}

// Get variance from Welford state
__device__ __forceinline__ float welford_variance(const WelfordState& state) {
    return (state.count > 1) ? state.M2 / state.count : 0.0f;
}

// Segmented Reduction (for MHKD gradient accumulation)

// Reduce values within segments of size segment_size
// Used when head_dim > 1 and multiple channels map to same head
template<int BLOCK_SIZE>
__device__ void segmented_reduce_add(
    float* output,           // Output array (one per segment)
    float val,               // Value to reduce
    int idx,                 // Global index of this thread
    int segment_size,        // Size of each segment (head_dim)
    float* shared_mem
) {
    // Determine which segment this thread belongs to
    int segment_id = idx / segment_size;
    int local_id = idx % segment_size;

    // Store value to shared memory
    int tid = threadIdx.x;
    shared_mem[tid] = val;
    __syncthreads();

    // Reduce within segment
    // This assumes segment_size is power of 2 and divides BLOCK_SIZE
    for (int stride = segment_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // First thread in each segment writes to output
    if (local_id == 0) {
        atomicAdd(&output[segment_id], shared_mem[tid]);
    }
}

// Cross-Lane Operations

// Get value from a specific lane in the warp
__device__ __forceinline__ float read_lane(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// Rotate values within warp (shift left with wrap)
__device__ __forceinline__ float warp_rotate_left(float val, int shift) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int src_lane = (lane_id + shift) % WARP_SIZE;
    return __shfl_sync(0xffffffff, val, src_lane);
}

// Prefix sum (inclusive scan) within warp
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    int lane_id = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int delta = 1; delta < WARP_SIZE; delta *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, delta);
        if (lane_id >= delta) {
            val += n;
        }
    }

    return val;
}

// Exclusive prefix sum within warp
__device__ __forceinline__ float warp_exclusive_scan(float val) {
    float inclusive = warp_inclusive_scan(val);
    return inclusive - val;
}

}  // namespace cdssm
