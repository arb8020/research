from problem import Machine, DebugInfo, Instruction

"""
VECTOR (SIMD) VLIW EXAMPLE
==========================
Goal: load 8 values from mem[0:8], add mem[8:16] to them, store back to mem[0:8]

Lessons learned:
1. vload/vstore operate on VLEN=8 contiguous elements
2. vload addr is STILL indirect: ("vload", dest, addr) loads from mem[scratch[addr]]
   - WRONG: ("vload", 0, 0) - tries to load from mem[scratch[0]] but scratch[0] is uninitialized
   - RIGHT: first ("const", 99, 0), then ("vload", 0, 99) - loads from mem[0]
3. vstore arg order: ("vstore", addr, src) - addr is scratch loc holding mem address, src is scratch loc holding data
   - WRONG: ("vstore", 0, 99) - writes scratch[99:107] to mem[scratch[0]]
   - RIGHT: ("vstore", 99, 0) - writes scratch[0:8] to mem[scratch[99]]
4. valu ops work on 8 elements: ("+", dest, a1, a2) does scratch[dest+i] = scratch[a1+i] + scratch[a2+i]

Graph representation of this program:
  const(0) ──→ vload ──→
                         add ──→ vstore
  const(8) ──→ vload ──→

6 nodes in graph, but only 4 cycles because independent ops can be packed.
"""

# VLEN is 8

# initialize mem
# load 8 values from memory into scratch (vload)
# add them together (valu)
# store them back (vstore)



mem = list(range(8)) + [1] * 8


my_program = [
    # scratch[99] = 0, scratch[100] = 8
    {"load": [("const", 99, 0), ("const", 100, 8)]}, 
    # load mem[scratch[99]] -> mem[0] to scratch[0:8] 
    {"load": [("vload", 0, 99), ("vload", 8, 100)]},

    {"valu": [("+", 0, 0, 8)]}, # vector add in place

    {"store": [("vstore", 99, 0)]},  # write from scratch to mem, overwrite original mem
]

machine = Machine(
    mem,
    my_program,
    DebugInfo(scratch_map={}),
)

machine.run()
print(f"scratch[0:16] = {machine.cores[0].scratch[0:16]}")
print(f"mem[0:16] = {machine.mem[0:16]}")

print(f"mem[0] = {machine.mem[0]}")
print(f"cycles: {len(list(my_program))}")

""" ISA Reference:

SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

LOAD ops:
  ("load", dest, addr)        - scratch[dest] = mem[scratch[addr]]
  ("vload", dest, addr)       - scratch[dest:dest+8] = mem[scratch[addr]:scratch[addr]+8]
  ("const", dest, val)        - scratch[dest] = val

STORE ops:
  ("store", addr, src)        - mem[scratch[addr]] = scratch[src]
  ("vstore", addr, src)       - mem[scratch[addr]:+8] = scratch[src:src+8]

ALU ops (dest, a1, a2):
  ("+", dest, a1, a2)         - scratch[dest] = scratch[a1] + scratch[a2]
  ("*", dest, a1, a2)         - scratch[dest] = scratch[a1] * scratch[a2]
  ... and: -, //, %, ^, &, |, <<, >>, <, ==

VALU ops (operates on 8 elements):
  ("vbroadcast", dest, src)   - scratch[dest:dest+8] = [scratch[src]] * 8
  (op, dest, a1, a2)          - scratch[dest+i] = scratch[a1+i] op scratch[a2+i] for i in 0..7

FLOW ops:
  ("select", dest, cond, a, b) - scratch[dest] = scratch[a] if scratch[cond] else scratch[b]
  ("vselect", dest, cond, a, b) - vector version
  ("cond_jump", cond, addr)   - if scratch[cond]: pc = addr
  ("jump", addr)              - pc = addr
  ("halt",)                   - stop execution
"""
