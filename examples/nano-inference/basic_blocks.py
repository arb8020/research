from dataclasses import dataclass
from typing import List

# @dataclass
# class Block:

@dataclass
class BlockManagerState:
    block_size: int
    n_blocks: int
    block_table: List[int]


@dataclass
class Sequence:
    tokens: List[int]



def allocate(n_tokens: int, bms: BlockManagerState) -> int:
    
    need_blocks = (n_tokens + bms.block_size-1) // bms.block_size # ceil div
    
    allocated_ids = []

    for i in range(bms.n_blocks):
        if bms.block_table[i] == 0:
            bms.block_table[i] = 1
            need_blocks -= 1
            allocated_ids += [i]
            if need_blocks == 0:
                break

    return allocated_ids
   

def deallocate(block_ids: List[int], bms: BlockManagerState) -> None:

    block_table = bms.block_table
    for id in block_ids:
        block_table[id] = 0

    return None 
        

def char_tokenize(prompt_str: str) -> List[int]:
    tokens = [ord(char) for char in prompt_str]
    return tokens

n_blocks = 64
bms = BlockManagerState(block_size = 8, n_blocks = n_blocks, block_table = ([0]*n_blocks))
initial_prompt = "hello world"
tokenized_prompt = char_tokenize(initial_prompt)
n_prefill = len(tokenized_prompt) 

allocated_ids = allocate(n_prefill, bms)
deallocate(allocated_ids, bms)

