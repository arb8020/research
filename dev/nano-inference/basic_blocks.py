from dataclasses import dataclass
from typing import List

# @dataclass
# class Block:

@dataclass
class BlockManagerState:
    block_size: int
    n_blocks: int
    free_blocks: List[int] # stack


@dataclass
class Sequence:
    tokens: List[int]


def can_allocate(n_tokens: int, bms: BlockManagerState) -> bool:
    need_blocks = (n_tokens + bms.block_size-1) // bms.block_size # ceil div
    n_free_blocks = len(bms.free_blocks)
    
    if need_blocks > n_free_blocks:
        return False

    return True

def allocate(n_tokens: int, bms: BlockManagerState) -> int:

    assert n_tokens > 0, "n tokens should be > 0"
    
    need_blocks = (n_tokens + bms.block_size-1) // bms.block_size # ceil div

    assert len(bms.free_blocks) >= need_blocks, "not enough free blocks"
    allocated_ids = []

    for i in range(need_blocks):
        freed_block_id = bms.free_blocks.pop()
        allocated_ids += [freed_block_id]

    assert len(allocated_ids) == need_blocks 

    return allocated_ids
   

def deallocate(block_ids: List[int], bms: BlockManagerState) -> None:

    assert len(block_ids) > 0

    for id in block_ids:
        bms.free_blocks.append(id)

    return None 
        

def char_tokenize(prompt_str: str) -> List[int]:
    tokens = [ord(char) for char in prompt_str]
    return tokens


def main():
    
    n_blocks = 64
    bms = BlockManagerState(block_size = 8, n_blocks = n_blocks, free_blocks = list(range(n_blocks)))
    
    initial_prompt = "hello world"
    max_tokens = 8
    prompt_tokens = char_tokenize(initial_prompt)
    
    sequence = prefill(prompt_tokens, bms)

    for i in range(max_tokens):
        new_token = model_generate(prefill_result)
        append_token(new_token, sequence, bms)

    cleanup(sequence, bms)

