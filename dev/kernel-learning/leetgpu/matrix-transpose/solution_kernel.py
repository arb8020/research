"""
Matrix Transpose

Difficulty: EASY
Challenge ID: 3
Language: cute

Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The
  transpose of a matrix switches its rows and columns. Given a matrix \(A\) of dimensions \(rows \times cols\), the transpose \(A^T\) will have dimensions \(cols \times rows\). All matrices are stored in row-major format.


Implementation Requirements

  Use only native features (external libraries are not permitted)
  The solve function signature must remain unchanged
  The final result must be stored in the matrix output


Example 1:
Input: 2×3 matrix
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0
\end{bmatrix}
\]

Output: 3×2 matrix
\[
\begin{bmatrix}
1.0 & 4.0 \\
2.0 & 5.0 \\
3.0 & 6.0
\end{bmatrix}
\]

Example 2:
Input: 3×1 matrix
\[
\begin{bmatrix}
1.0 \\
2.0 \\
3.0
\end{bmatrix}
\]

Output: 1×3 matrix
\[
\begin{bmatrix}
1.0 & 2.0 & 3.0
\end{bmatrix}
\]

Constraints

  1 ≤ rows, cols ≤ 8192
  Input matrix dimensions: rows × cols
  Output matrix dimensions: cols × rows
"""

import cutlass.cute as cute


# input, output are tensors on the GPU
@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    kernel = transpose(input, output, rows, cols)

    n_ops = rows * cols
    n_tpb = 32  # 1024

    ceil_div = lambda n, d: (n + d - 1) // d

    grid_blocks_x = ceil_div(rows, n_tpb)
    grid_blocks_y = ceil_div(cols, n_tpb)

    kernel.launch(grid=(grid_blocks_x, grid_blocks_y, 1), block=(n_tpb, n_tpb, 1))

    pass


@cute.kernel
def transpose(input: cute.Tensor, output: cute.Tensor, rows: cute.Int32, cols: cute.Int32):
    # what is a transpose
    # A[i,j] = B[j,i]

    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    j = bidx * bdimx + tidx
    i = bidy * bdimy + tidy

    if i < rows and j < cols:
        output[j, i] = input[i, j]

    pass
