from problem import Machine, DebugInfo, Instruction

"""
SCALAR VLIW EXAMPLE
===================
Goal: compute (42 + 25) * 2 = 134, store to mem[0]

Lessons learned:
1. All addresses in instructions refer to SCRATCH, not memory directly
2. Memory access is INDIRECT: to access mem[0], first load 0 into scratch, then use that scratch loc
3. store format is ("store", addr_scratch_loc, value_scratch_loc) - addr comes from scratch[addr_scratch_loc]
4. VLIW packing: one instruction bundle can use ALL engines simultaneously {"load": [...], "alu": [...]}
5. Multiple slots per engine: {"load": [op1, op2]} runs both in same cycle (up to SLOT_LIMITS)
6. Data dependencies: can't read a value in the same cycle it's written (writes happen at end of cycle)

Original: 7 cycles (one op per instruction)
Optimized: 4 cycles (packed independent ops)
"""

# VLEN is 8

# initialize mem
mem = [0] * 16
my_program = [
    {"load": [("const", 0, 42), ("const", 1, 25)]}, 
    {"load": [("const", 2, 2), ("const", 5, 0)], "alu": [("+", 3, 0, 1)]},  
    {"alu": [("*", 4, 3, 2)]},  # scratch[4] = 134
    {"store": [("store", 5, 4)]},  # mem[0] = scratch[4]
]

machine = Machine(
    mem,
    my_program,
    DebugInfo(scratch_map={}),
)

machine.run()
print(f"mem[0] = {machine.mem[0]}")
print(f"cycles: {len(list(my_program))}")

"""
notes: 
SLOT_LIMITS = {
      "alu": 12,
      "valu": 6,
      "load": 2,
      "store": 2,
      "flow": 1,
      "debug": 64,
  }

"""
