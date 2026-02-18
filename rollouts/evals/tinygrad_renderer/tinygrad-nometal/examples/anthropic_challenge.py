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

    inst = []
    for i, u in enumerate(uops):
      # render UOps to instructions
      match u.op:
        case Ops.SINK:
          inst.append({"flow": [("halt",)]})
        case Ops.CONST:
          inst.append({"load": [("const", r[u], u.arg)]})
        case Ops.GEP:
          # a GEP is just an alias to a special register in the vector
          r[u] = r[u.src[0]] + u.arg[0]
        case Ops.VECTORIZE:
          if all(s == u.src[0] for s in u.src):
            # if all sources are the same, we can broadcast
            inst.append({"valu": [("vbroadcast", r[u], r[u.src[0]])]})
          else:
            # this is a copy into a contiguous chunk of registers
            inst.extend({"flow": [("add_imm", r[u]+i, r[s], 0)]} for i,s in enumerate(u.src) if r[s] != r[u]+i)
        case Ops.LOAD:
          op = "vload" if u.dtype.count > 1 else "load"
          inst.append({"load": [(op, r[u], r[u.src[0]])]})
        case Ops.STORE:
          op = "vstore" if u.src[1].dtype.count > 1 else "store"
          inst.append({"store": [(op, r[u.src[0]], r[u.src[1]])]})
        case Ops.MULACC:
          assert u.dtype.count == 8
          inst.append({"valu": [("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])]})
        case Ops.WHERE:
          assert u.dtype.count == 8
          inst.append({"flow": [("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])]})
        case _ if u.op in self.code_for_op:
          cat = "valu" if u.dtype.count > 1 else "alu"
          inst.append({cat: [(self.code_for_op[u.op], r[u], r[u.src[0]], r[u.src[1]])]})
        case _:
          raise NotImplementedError(f"unhandled op {u.op}")
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
