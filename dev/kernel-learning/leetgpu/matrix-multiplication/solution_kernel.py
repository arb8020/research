"""
Matrix Multiplication

Difficulty: EASY
Challenge ID: 2
Language: cute

Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU.
  Given matrix \(A\) of dimensions \(M \times N\) and matrix \(B\) of dimensions \(N \times K\), compute
  the product matrix \(C = A \times B\), which will have dimensions \(M \times K\).
  All matrices are stored in row-major format.


Implementation Requirements

  Use only native features (external libraries are not permitted)
  The solve function signature must remain unchanged
  The final result must be stored in matrix C


Example 1:

Input:
Matrix \(A\) (\(2 \times 2\)):
\[
\begin{bmatrix}
1.0 & 2.0 \\
3.0 & 4.0
\end{bmatrix}
\]
Matrix \(B\) (\(2 \times 2\)):
\[
\begin{bmatrix}
5.0 & 6.0 \\
7.0 & 8.0
\end{bmatrix}
\]
Output:
Matrix \(C\) (\(2 \times 2\)):
\[
\begin{bmatrix}
19.0 & 22.0 \\
43.0 & 50.0
\end{bmatrix}
\]


Example 2:

Input:
Matrix \(A\) (\(1 \times 3\)):
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0
\end{bmatrix}
\]
Matrix \(B\) (\(3 \times 1\)):
\[
\begin{bmatrix}
4.0 \\
5.0 \\
6.0
\end{bmatrix}
\]
Output:
Matrix \(C\) (\(1 \times 1\)):
\[
\begin{bmatrix}
32.0
\end{bmatrix}
\]


Constraints

  1 &le; M, N, K &le; 8192
  Performance is measured with M = 8192, N = 6144, K = 4096
"""

import cutlass.cute as cute


@cute.jit
def solve(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32
):
    kernel = matmul(A, B, C, M, N, K)

    block_threads = 32  # 32 x 32 for 2D grid, 1024 tpb

    matmul_ops = M * K  # C[i,j] = A[i,k] * B[k,j]

    # need to set up launch config as 2D now
    # based on M, K

    ceil_div = lambda n, d: (n + d - 1) // d

    grid_blocks_x = ceil_div(K, block_threads)
    grid_blocks_y = ceil_div(M, block_threads)

    kernel.launch(
        grid=(grid_blocks_x, grid_blocks_y, 1),
        block=(block_threads, block_threads, 1),
    )

    pass


@cute.kernel
def matmul(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cute.Int32, N: cute.Int32, K: cute.Int32
):
    # 2D launch config

    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    # we want to index into the matrix as if it were flat

    j = bidx * bdimx + tidx
    i = bidy * bdimy + tidy

    if i < M and j < K:
        acc_dot = 0.0

        for n in range(N):
            # n is shared dim between A: [M,N] B: [N,K]

            # A[i,k] -> A is [M,N]
            # B[k,j] -> B is [N,K]

            acc_dot += A[i, n] * B[n, j]

        C[i, j] = acc_dot

    pass
