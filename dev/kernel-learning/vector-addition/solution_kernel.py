# Vector Addition
#
# Difficulty: EASY
# Challenge ID: 1
# Language: cute
#
# Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU.
#   The program should take two input vectors of equal length and produce a single output vector containing their sum.
#
#
# Implementation Requirements
#
#   External libraries are not permitted
#     The solve function signature must remain unchanged
#     The final result must be stored in vector C
#
#
# Example 1:
#
# Input:  A = [1.0, 2.0, 3.0, 4.0]
#         B = [5.0, 6.0, 7.0, 8.0]
# Output: C = [6.0, 8.0, 10.0, 12.0]
#
#
# Example 2:
#
# Input:  A = [1.5, 1.5, 1.5]
#         B = [2.3, 2.3, 2.3]
# Output: C = [3.8, 3.8, 3.8]
#
#
# Constraints
#
#
#   Input vectors A and B have identical lengths
#   1 &le; N &le; 100,000,000
#

import cutlass.cute as cute


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    pass
