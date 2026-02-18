from tinygrad import Tensor, dtypes, Context, getenv, UOp, fetch
from tinygrad.uop.ops import Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps

# ************************* implementation of the problem ************************

def myhash(a: Tensor) -> Tensor:
  a = (a + 0x7ED55D16) + (a << 12)
  a = (a ^ 0xC761C23C) ^ (a >> 19)
  a = (a + 0x165667B1) + (a << 5)
  a = (a + 0xD3A2646C) ^ (a << 9)
  a = (a + 0xFD7046C5) + (a << 3)
  a = (a ^ 0xB55A4F09) ^ (a >> 16)
  return a

def select_with_where_tree(values: Tensor, relative_idx: Tensor) -> Tensor:
  n = values.shape[0]
  if n == 1: return values[0].expand(relative_idx.shape)

  mid = n // 2
  left = select_with_where_tree(values[:mid], relative_idx)
  right = select_with_where_tree(values[mid:], relative_idx - mid)

  go_left = relative_idx < mid
  return go_left.where(left, right)

def tree_traversal(forest: Tensor, val: Tensor, height: int, rounds: int, where_tree_threshold=3) -> Tensor:
  # All walkers start at idx=0
  idx = Tensor.zeros(val.shape, device=val.device, dtype=dtypes.uint32)

  for r in range(rounds):
    level = r % (height + 1)
    level_start = (1 << level) - 1
    level_size = 1 << level

    if level == 0:
      # At root (level 0), all walkers are at idx=0
      # No gather needed, just broadcast the root value
      node_val = forest[0].expand(val.shape)
      idx = idx * 0  # Reset to 0
    elif level <= where_tree_threshold:
      # Small level: use where-tree
      level_values = forest[level_start : level_start + level_size]
      relative_idx = (idx - level_start)
      node_val = select_with_where_tree(level_values, relative_idx)
    else:
      # Large level: use gather
      node_val = forest.gather(0, idx)

    val = myhash(val ^ node_val)
    idx = (idx << 1) + (1 + (val & 1))

    # No wrap check needed! At round 10 (level becomes 0), we reset idx above.

  return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))

# ************************* renderer for VLIW machine *************************

def loop_unrolling(sink:UOp):
  rng = [x for x in sink.toposort() if x.op is Ops.RANGE]
  if len(rng) == 0: return None
  print(f"unrolling loop with size {rng[0].vmax+1}")
  unrolled_sinks = [sink.substitute({rng[0]:rng[0].const_like(i)}).src[0] for i in range(rng[0].vmax+1)]
  return UOp.sink(*unrolled_sinks, arg=sink.arg)

global_addrs = []
vliw_prepare = PatternMatcher([
  # loop unrolling (should be a part of tinygrad)
  (UPat(Ops.SINK, name="sink"), loop_unrolling),
  # cast is fake
  (UPat(Ops.CAST, name="c"), lambda c: c.src[0]),
  # rewrites to hardcode the addresses in memory
  (UPat(Ops.PARAM, name="dg"), lambda dg: UOp.const(dtypes.uint, global_addrs[dg.arg])),
  # INDEX is just plus
  (UPat(Ops.INDEX, name="i"), lambda i: i.src[0]+i.src[1]),
])+symbolic

class VLIWRenderer(Renderer):
  has_local = False  # TODO: this should be the default / cleaned up
  # this says this backend supports MULACC + more. decompositions uses this
  code_for_op: dict = {Ops.MULACC: None, Ops.ADD: "+", Ops.MUL: "*",
                       Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|",
                       Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
  # this matcher runs while still in graph form
  pre_matcher = vliw_prepare

  def render(self, uops:list[UOp]):

    # TODO: this is a minimal renderer. for low cycle count, make it good
    # to get speed, you need to add VLIW packing
    # to get under 1536 regs, you need to add a register allocator
    # we left the fun parts to you

    print(f"rendering with {len(uops)} uops")

    # === NEW: liveness-based register allocator ===
    # Step 1: find last use of each uop
    # GEP is special: it aliases its source, so the source's lifetime extends to GEP's last use
    gep_source: dict[UOp, UOp] = {}  # gep -> original vector
    for u in uops:
      if u.op is Ops.GEP:
        src = u.src[0]
        # follow chain of GEPs to find original vector
        while src in gep_source:
          src = gep_source[src]
        gep_source[u] = src

    last_use: dict[UOp, int] = {}
    for i, u in enumerate(uops):
      for src in u.src:
        # if this is a GEP, extend the original vector's lifetime
        actual_src = gep_source.get(src, src)
        last_use[actual_src] = i

    # Step 2: allocate registers, freeing after last use
    # Split register space: 0-1023 for vectors (aligned to 8), 1024-1535 for scalars
    free_regs_8 = list(range(0, 1024, 8))  # vector registers (128 available)
    free_regs_1 = list(range(1024, 1536))  # scalar registers (512 available)
    r: dict[UOp, int] = {}
    freed: set[UOp] = set()  # track what we've freed to avoid double-free
    max_reg_used = 0

    for i, u in enumerate(uops):
      assert u.dtype.count in (1,8), "dtype count must be 1 or 8"

      # allocate register for this uop (if it needs one)
      if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}:
        if u.dtype.count == 8:
          assert free_regs_8, f"out of vector registers at uop {i}"
          reg = free_regs_8.pop(0)
        else:
          assert free_regs_1, f"out of scalar registers at uop {i}"
          reg = free_regs_1.pop(0)
        r[u] = reg
        max_reg_used = max(max_reg_used, reg + u.dtype.count)

      # free registers for sources whose last use was this instruction
      for src in u.src:
        # if this is a GEP, check the original vector's last use
        actual_src = gep_source.get(src, src)
        if last_use.get(actual_src) == i and actual_src in r and actual_src not in freed:
          freed.add(actual_src)
          if actual_src.dtype.count == 8:
            free_regs_8.append(r[actual_src])
          else:
            free_regs_1.append(r[actual_src])

    print(f"max register used: {max_reg_used}")
    # === END NEW ===

    # # === OLD: dumb register allocator ===
    # reg, r = 0, {}
    # for u in uops:
    #   assert u.dtype.count in (1,8), "dtype count must be 1 or 8"
    #   if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}:
    #     r[u] = reg
    #     reg += u.dtype.count
    # # === END OLD ===

    # === Step 1: render UOps to (engine, slot) pairs ===
    # Each slot is (engine, slot_tuple, producing_uop, depends_on_uops)
    raw_slots: list[tuple[str, tuple, UOp, list[UOp]]] = []
    for i, u in enumerate(uops):
      match u.op:
        case Ops.SINK:
          raw_slots.append(("flow", ("halt",), u, list(u.src)))
        case Ops.CONST:
          raw_slots.append(("load", ("const", r[u], u.arg), u, []))
        case Ops.GEP:
          r[u] = r[u.src[0]] + u.arg[0]
        case Ops.VECTORIZE:
          if all(s == u.src[0] for s in u.src):
            raw_slots.append(("valu", ("vbroadcast", r[u], r[u.src[0]]), u, list(u.src)))
          else:
            for vi, s in enumerate(u.src):
              if r[s] != r[u] + vi:
                raw_slots.append(("flow", ("add_imm", r[u]+vi, r[s], 0), u, [s]))
        case Ops.LOAD:
          op = "vload" if u.dtype.count > 1 else "load"
          raw_slots.append(("load", (op, r[u], r[u.src[0]]), u, list(u.src)))
        case Ops.STORE:
          op = "vstore" if u.src[1].dtype.count > 1 else "store"
          raw_slots.append(("store", (op, r[u.src[0]], r[u.src[1]]), u, list(u.src)))
        case Ops.MULACC:
          assert u.dtype.count == 8
          raw_slots.append(("valu", ("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]), u, list(u.src)))
        case Ops.WHERE:
          assert u.dtype.count == 8
          raw_slots.append(("flow", ("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]), u, list(u.src)))
        case _ if u.op in self.code_for_op:
          cat = "valu" if u.dtype.count > 1 else "alu"
          raw_slots.append((cat, (self.code_for_op[u.op], r[u], r[u.src[0]], r[u.src[1]]), u, list(u.src)))
        case _:
          raise NotImplementedError(f"unhandled op {u.op}")

    # === Step 2: greedy VLIW packing ===
    SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

    # Extract register reads/writes from each slot
    def get_slot_regs(engine: str, slot: tuple) -> tuple[set[int], set[int]]:
      """Returns (reads, writes) as sets of register numbers."""
      reads, writes = set(), set()
      match (engine, slot[0]):
        case ("flow", "halt"): pass
        case ("load", "const"):
          writes.add(slot[1])
        case ("valu", "vbroadcast"):
          writes.update(range(slot[1], slot[1]+8))
          reads.add(slot[2])
        case ("flow", "add_imm"):
          writes.add(slot[1])
          reads.add(slot[2])
        case ("load", "load"):
          writes.add(slot[1])
          reads.add(slot[2])
        case ("load", "vload"):
          writes.update(range(slot[1], slot[1]+8))
          reads.add(slot[2])
        case ("store", "store"):
          reads.add(slot[1])
          reads.add(slot[2])
        case ("store", "vstore"):
          reads.add(slot[1])
          reads.update(range(slot[2], slot[2]+8))
        case ("valu", "multiply_add"):
          writes.update(range(slot[1], slot[1]+8))
          reads.update(range(slot[2], slot[2]+8))
          reads.update(range(slot[3], slot[3]+8))
          reads.update(range(slot[4], slot[4]+8))
        case ("flow", "vselect"):
          writes.update(range(slot[1], slot[1]+8))
          reads.update(range(slot[2], slot[2]+8))
          reads.update(range(slot[3], slot[3]+8))
          reads.update(range(slot[4], slot[4]+8))
        case ("alu", _) | ("valu", _):
          if engine == "valu":
            writes.update(range(slot[1], slot[1]+8))
            reads.update(range(slot[2], slot[2]+8))
            reads.update(range(slot[3], slot[3]+8))
          else:
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])
      return reads, writes

    slot_regs = [get_slot_regs(e, s) for e, s, _, _ in raw_slots]

    # Build uop-based dependencies (for things like SINK that depend on STOREs)
    uop_to_slots: dict[UOp, list[int]] = {}
    for i, (_, _, producing_uop, _) in enumerate(raw_slots):
      uop_to_slots.setdefault(producing_uop, []).append(i)

    def get_producing_slots(uop: UOp) -> list[int]:
      while uop in gep_source:
        uop = gep_source[uop]
      return uop_to_slots.get(uop, [])

    # Build data dependencies combining:
    # 1. Uop-based: slot depends on its source uops' slots
    # 2. Register RAW: slot i reads what slot j writes
    # 3. Register WAR (anti-dep): slot i writes what slot j reads
    deps: list[set[int]] = [set() for _ in raw_slots]
    anti_deps: list[set[int]] = [set() for _ in raw_slots]

    for i, (engine, slot, producing_uop, dep_uops) in enumerate(raw_slots):
      reads_i, writes_i = slot_regs[i]

      # Uop-based dependencies
      for src in dep_uops:
        deps[i].update(get_producing_slots(src))

      # Register-based dependencies with earlier slots
      for j in range(i):
        reads_j, writes_j = slot_regs[j]
        # RAW: i reads what j writes
        if reads_i & writes_j:
          deps[i].add(j)
        # WAR: i writes what j reads (anti-dependency)
        if writes_i & reads_j:
          anti_deps[i].add(j)

    # greedy scheduling
    scheduled = [False] * len(raw_slots)
    completed = set()  # indices of slots whose results are available
    inst = []

    USE_VLIW_PACKING = getenv("VLIW", 1)

    if USE_VLIW_PACKING:
      while not all(scheduled):
        bundle: dict[str, list] = {}
        slot_counts = {k: 0 for k in SLOT_LIMITS}
        scheduled_this_cycle = []

        for i, (engine, slot, u, _) in enumerate(raw_slots):
          if scheduled[i]:
            continue
          # check data dependencies (RAW) - producer must be done
          if not deps[i].issubset(completed):
            continue
          # check anti-dependencies (WAR) - earlier readers must be done before we overwrite
          if not anti_deps[i].issubset(completed):
            continue
          # check slot limit
          if slot_counts[engine] >= SLOT_LIMITS[engine]:
            continue
          # pack it
          if engine not in bundle:
            bundle[engine] = []
          bundle[engine].append(slot)
          slot_counts[engine] += 1
          scheduled[i] = True
          scheduled_this_cycle.append(i)

        assert bundle, "deadlock: no progress made"
        inst.append(bundle)
        # results are available next cycle
        completed.update(scheduled_this_cycle)

      print(f"packed {len(raw_slots)} slots into {len(inst)} bundles")
    else:
      # no packing - one slot per bundle (original behavior)
      for engine, slot, u, _ in raw_slots:
        inst.append({engine: [slot]})
      print(f"no packing: {len(inst)} bundles")
    return repr(inst)

# ************************* test and render *************************

import sys, types
PROBLEM_URL = "https://raw.githubusercontent.com/anthropics/original_performance_takehome/refs/heads/main/tests/frozen_problem.py"
sys.modules["problem"] = problem = types.ModuleType("problem")
exec(fetch(PROBLEM_URL).read_text(), problem.__dict__)

if __name__ == "__main__":
  batch_size = getenv("BS", 256)
  height = 10
  rounds = getenv("ROUNDS", 16)

  # build problem
  tree = problem.Tree.generate(height)
  inp = problem.Input.generate(tree, batch_size, rounds)
  mem = problem.build_mem_image(tree, inp)
  global_addrs.extend([mem[6], mem[6], mem[4]])  # output, input, forest

  # *** verify the kernel in tinygrad compared to reference ***

  forest_t = Tensor(tree.values, dtype=dtypes.uint32)
  val_t = Tensor(inp.values, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    # verify on normal tinygrad device
    with Context(PCONTIG=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    problem.reference_kernel(tree, inp)
    assert val_out == inp.values
    print("verification passed")

  # *** render to device ***

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, SPEC=0):
    out = tree_traversal(forest_t, val_t, height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  # *** run on Machine and compare ***

  # NOTE: the scratch size needs to be reduced to 1536 when you have a register allocator
  src = eval(prg.src)
  max_regs = max(t[1] for instr in src for v in instr.values() for t in v if len(t) > 1) + 1
  print(f"{max_regs:5d} regs used" + ("" if max_regs <= 1536 else "       <-- WARNING: TOO MANY REGISTERS, MUST BE <= 1536"))
  machine = problem.Machine(mem, src, problem.DebugInfo(scratch_map={}), n_cores=1, trace=False, scratch_size=1536)
  machine.run()
  print(f"ran for {machine.cycle:5d} cycles" + ("" if machine.cycle <= 1363 else "  <-- EVEN CLAUDE GOT 1363"))

  # compare to reference
  ref_mem = mem.copy()
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
  print("compare passed!")
