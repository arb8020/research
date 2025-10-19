Variable names should be concise and informative. For a tensor, nothing is more informative than how many dimensions it has, and what those dimensions represent.

Get Noam Shazeer’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
We have been keeping this convention at Character.AI since 2022. Give it a try and let me know if you feel saner:

Designate a system of single-letter names for logical dimensions, e.g. B for batch size, L for sequence length, etc., and document it somewhere in your file/project/codebase
When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. input_token_id_BL for a two-dimensional tensor with batch and length dimensions.
That’s all. You can use shape suffixes with torch, JAX, whatever. See the example below.

"""
B: batch
L: layers
T: sequence length (query)
S: sequence length (key value)
V: vocab size
D: d_model
F: MLP hidden dim
H: attn head dim
N: query heads
K: key value heads
G: q heads per kv head
"""
