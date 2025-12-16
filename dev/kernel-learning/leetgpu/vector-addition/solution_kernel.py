"""
## Problem Description

  Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU.
  The program should take two input vectors of equal length and produce a single output vector containing their sum.


Implementation Requirements

  External libraries are not permitted
    The solve function signature must remain unchanged
    The final result must be stored in vector C


Example 1:

Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]


Example 2:

Input:  A = [1.5, 1.5, 1.5]
        B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]


Constraints

  Input vectors A and B have identical lengths
  1 &le; N &le; 100,000,000
"""

import cutlass.cute as cute

# runs on host, launches the cute.kernel decorated functions
# we need this to define kernel launch configuration
# gpus run threads in parallel, rather than sequential
# instead of something like
# for i in range(N): C[i] = A[i] + B[i]
# where python figures out how to sequentially execute
# the gpu computes each 'i' in parallel
# each index gets its own thead
# so we need to tell it how many threads

# NVIDIA's GPU model makes a 3 level hierarchy
# threads are organized into a 3D block of threads
# blocks are organized into a 3D block of grids
# so we need to tell the kernel what grid configuration to launch
# ie: how many blocks and in what shape
# and what block configuration to launch
# ie: how many threads and in what shape


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    kernel = vec_add_1D(A, B, C, N)

    # we only need 1D
    # GPUs also execute threads in groups of 32 called warps
    # so the number of threads per block size should always be a multiple of 32 for best perf

    num_threads_per_block = 256

    # since we have N length for A and B, N/256 blocks

    num_blocks_per_grid = (N + num_threads_per_block - 1) // num_threads_per_block

    kernel.launch(
        grid=(num_blocks_per_grid, 1, 1),
        block=(num_threads_per_block, 1, 1),
    )

    pass


# now we write the kernel body
# what we need to do is really simple: C[i] = A[i] + B[i]

# since all threads execute in parallel
# we write the kernel as if it were from a single thread's perspective
# so each thread needs to know which part of the vector its doing
# how do we calculate 'i' for a given run of the kernel operation then?


@cute.kernel
def vec_add_1D(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, N: cute.Uint32):
    # each thread knows its position within a block, through 'thread_idx'
    # it also knows which its block within the grid, through 'block_idx'
    # we can also access the threads per block, through 'block_dim'
    # each of these are 3D, and we only need the x dimension

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    # so for a given index, say '732'
    # we need to multiply threads per block by the block index into the grid
    # and offset by the thread index into the block

    thread_idx = bdim * bidx + tidx

    if thread_idx < N:  # protect against out of range access
        gC[thread_idx] = gA[thread_idx] + gB[thread_idx]


# naming convention note:
# we write 'gA' instead of 'A' as a way to note that we are using a global memory tensor
# the global memory is the main GPU RAM that all threads can access
# we might later prefix with 's' for 'shared memory', shared in a thread block
# or 'r' for 'register memory', private to a single thread
