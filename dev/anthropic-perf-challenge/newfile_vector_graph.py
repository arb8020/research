from dataclasses import dataclass
from typing import List, Any, Dict, Tuple

Instruction = Tuple[str, List[tuple]]
Cycle = Dict[str, List[tuple]]

@dataclass(frozen=True)
class Node:
    op: str
    inputs: tuple['Node', ...]
    arg: Any = None


addr0 = Node("const", (), 0)
addr8 = Node("const", (), 8)
load1 = Node("vload", (addr0,))
load2 = Node("vload", (addr8,))
add2 = Node("valu", (load1, load2))
store = Node("vstore", (addr0, add2))

graph = (
    addr0, addr8, load1, load2, add2, store
)

def lower(node: Node, visited: Dict[Node, int], scratch_idx: int) -> Tuple[Instruction, int]:
    match node.op:
        case "const":
            instr = ("load", [("const", scratch_idx, node.arg)])
        case "vload":
            instr = ("load", [("vload", scratch_idx, visited[node.inputs[0]])])
        case "valu":
            instr = ("valu", [("+", scratch_idx, visited[node.inputs[0]], visited[node.inputs[1]])])
        case "vstore":
            instr = ("store", [("vstore", visited[node.inputs[0]], visited[node.inputs[1]])])
        case _:
            instr = ("???", [])
    return instr, scratch_idx + 1

def visit(node: Node, visited: Dict[Node, int] | None = None, scratch_idx: int = 0) -> Tuple[List[Instruction], Dict[Node, int], int]:
    if visited is None:
        visited = {}
    if node in visited:
        return [], visited, scratch_idx

    instrs = []
    for inp in node.inputs:
        inp_instrs, visited, scratch_idx = visit(inp, visited, scratch_idx)
        instrs.extend(inp_instrs)

    instr, scratch_idx = lower(node, visited, scratch_idx)
    instrs.append(instr)
    visited[node] = scratch_idx - 1

    return instrs, visited, scratch_idx

SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

def get_reads_writes(instr: Instruction) -> Tuple[set, set]:
    engine, slots = instr
    reads, writes = set(), set()
    for slot in slots:
        op = slot[0]
        if op == "const":
            writes.add(slot[1])
        elif op in ("store", "vstore"):
            reads.update(slot[1:])
        else:
            writes.add(slot[1])
            reads.update(slot[2:])
    return reads, writes

def schedule(instrs: List[Instruction], node_map: Dict[Node, int]) -> List[Cycle]:
    remaining = list(instrs)
    available_scratches = set()
    cycles = []

    while remaining:
        cycle: Cycle = {}
        writes_this_cycle = set()
        still_remaining = []

        for instr in remaining:
            engine, slots = instr
            reads, writes = get_reads_writes(instr)

            deps_satisfied = reads <= available_scratches
            slot_count = len(cycle.get(engine, []))
            has_slot = slot_count + len(slots) <= SLOT_LIMITS.get(engine, 1)

            if deps_satisfied and has_slot:
                if engine not in cycle:
                    cycle[engine] = []
                cycle[engine].extend(slots)
                writes_this_cycle.update(writes)
            else:
                still_remaining.append(instr)

        available_scratches.update(writes_this_cycle)
        cycles.append(cycle)
        remaining = still_remaining

    return cycles

def render(root: Node) -> List[Cycle]:
    instrs, node_map, _ = visit(root)
    return schedule(instrs, node_map)


if __name__ == "__main__":
    instrs, node_map, final_idx = visit(store)
    print("Instructions (unscheduled):")
    for i in instrs:
        print(" ", i)
    print(f"\nNode -> scratch mapping: {len(node_map)} nodes")
    print(f"Final scratch_idx: {final_idx}")

    cycles = render(store)
    print(f"\nScheduled into {len(cycles)} cycles:")
    for i, cycle in enumerate(cycles):
        print(f"  Cycle {i}: {cycle}")
