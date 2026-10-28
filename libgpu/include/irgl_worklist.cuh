#pragma once
/*
 * Copyright (C) 2014--2016, The University of Texas at Austin.
 * Derived from worklist.h (GGC, Sreepathi Pai). See LICENSE.txt.
 *
 * Allocation-free, bounded worklist view for modern CUDA applications. Keeps
 * Worklist's atomic range reservation and Worklist2's cooperative scan/append,
 * without graph, ModernGPU, tracing, or host allocation dependencies.
 *
 * Bulk-synchronous contract: producers and consumers run in separate ordered
 * kernel invocations. The counter reserves storage; it is NOT a publication
 * flag for concurrent consumers. Reset only after all consumers have finished.
 * Capacity is fixed. Overflow is reported, never written out of bounds.
 */
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
namespace galois { namespace gpu {
struct WorklistView {
    int* items;
    int* count;
    int capacity;
    int* overflow;

    __device__ int nitems() const { return *count < capacity ? *count : capacity; }
    __device__ int push_range(int n) const {
        const int start = atomicAdd(count, n);
        if (start > capacity - n) { atomicExch(overflow, 1); return -1; }
        return start;
    }
    __device__ bool push(int item) const {
        const int start = push_range(1);
        if (start < 0) return false;
        items[start] = item;
        return true;
    }
    __device__ bool pop_id(int index, int& item) const {
        if (index < 0 || index >= nitems()) return false;
        item = items[index];
        return true;
    }
    // All block threads must participate. Ordering is stable within a block,
    // while order between blocks follows atomic range reservations.
    template<int Threads>
    __device__ void push_if(bool present, int item,
            typename cub::BlockScan<int, Threads>::TempStorage& scratch,
            int& shared_start) const {
        int rank, total;
        cub::BlockScan<int, Threads>(scratch).ExclusiveSum(int(present), rank, total);
        if (!threadIdx.x) shared_start = total ? push_range(total) : -1;
        __syncthreads();
        if (present && shared_start >= 0) items[shared_start + rank] = item;
        __syncthreads();
    }
};
}} // namespace galois::gpu
