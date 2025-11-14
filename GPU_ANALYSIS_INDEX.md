# Backend-Bench GPU Execution Analysis - Complete Index

Generated: 2025-11-12  
Based on: Complete source code exploration of backend-bench GPU execution paths

## Overview

This analysis package contains comprehensive documentation of how backend-bench executes GPU kernel code, including both local and remote (Modal) execution paths.

## Documents Included

### 1. **GPU_EXECUTION_ANALYSIS.md** (PRIMARY - 8.5KB)
Complete deep-dive analysis with all source code locations and detailed explanations.

**Contents:**
- Section 1: GPU Configuration (BackendBenchConfig)
- Section 2: Local vs Remote Execution Path
- Section 3: Local GPU Execution (run_code function)
- Section 4: Remote GPU Execution via Modal
- Section 5: GPU-Specific Code in Execution Path
- Section 6: Test Input/Output Device Placement
- Section 7: Triton Kernel Compilation and Execution
- Section 8: Triton Benchmarking
- Section 9: Complete GPU Execution Flow Diagram
- Section 10: Key Summary of GPU Usage
- Section 11: GPU-Related Code Snippets Reference
- Section 12: Execution Mode Comparison

**Use this for:** Complete understanding with code snippets, line numbers, and detailed explanations.

### 2. **GPU_EXECUTION_QUICK_REFERENCE.md** (QUICK LOOKUP - 5.2KB)
Quick reference guide with direct answers to all 7 questions.

**Contents:**
- Answer 1: How is the GPU configured?
- Answer 2: What happens when gpu != "local"?
- Answer 3: How does the compiled kernel actually run on GPU?
- Answer 4: Are test inputs/outputs on GPU or CPU?
- Answer 5: CUDA, torch.cuda, and GPU-specific code
- Answer 6: How does Triton kernel compilation work?
- Answer 7: Difference between local vs non-local GPU execution
- Code Locations Quick Reference Table
- Key Code Snippets

**Use this for:** Quick lookups and specific code examples.

### 3. **GPU_EXECUTION_SUMMARY.txt** (TEXT SUMMARY - 6.8KB)
Comprehensive text summary answering all 7 questions with no markdown.

**Contents:**
- Answers to all 7 questions with code examples
- File locations summary
- Key insights (7 bullet points)
- Detailed comparison table

**Use this for:** Pure text format, easy copying, comprehensive reference.

---

## Quick Navigation

### I need to understand...

**GPU Configuration & Decision Making**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 1 & 7)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 1 & 2)

**How Code Actually Executes on GPU**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 3)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 3, 4, 5)

**Test Data Placement**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 4)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 6)

**CUDA & torch.cuda Usage**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 5)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 5)

**Triton Compilation**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 6)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 7)

**Local vs Remote Differences**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Answer 7)
→ Or: GPU_EXECUTION_ANALYSIS.md (Section 2, 4, 12)

**Code Locations**
→ See: GPU_EXECUTION_QUICK_REFERENCE.md (Code Locations Table)
→ Or: GPU_EXECUTION_ANALYSIS.md (throughout with line numbers)

---

## Key Findings Summary

### 1. GPU Configuration
- Single `gpu` parameter controls execution mode
- Options: `"local"` (local GPU) or `"T4"/"L4"/"A100"/"H100"/"H200"/"B200"` (Modal cloud)
- Default: `"T4"` (Modal)
- Located in: `src/config.py`

### 2. Execution Paths
- **Local** (`gpu="local"`): Direct function call via thread pool
  - Location: `src/utils.py:run_code()`
  - No scheduling, unreliable benchmarks

- **Remote** (`gpu != "local"`): Remote Modal function call
  - Location: `src/modal_runner.py:eval_code()`
  - Proper scheduling, reliable benchmarks
  - NVIDIA CUDA 12.8 Docker container

### 3. GPU Decision Point
```python
# In CodeEvaluator.__init__
if self._gpu == "local":
    self.callable = run_code
else:
    self.callable = lambda **kwargs: fn.remote(**kwargs)
```
**File:** `src/code_evaluator.py` Lines 124-140

### 4. Kernel Compilation
- Via `importlib.util.exec_module()`
- Triton JIT compilation happens during module execution
- Detected by `"@triton.jit"` pattern in code
- Automatically adds necessary imports

**File:** `BackendBench/utils.py:compile_kernel_from_string()` Lines 410-443

### 5. GPU Execution
- Test tensors created with `device="cuda"` by default
- Falls back to CPU if `torch.cuda.is_available()` returns False
- Both reference and implementation receive same tensor data
- No explicit device transfers

**File:** `BackendBench/utils.py:_deserialize_tensor()` Lines 100-113

### 6. GPU Availability Checking
- `torch.cuda.is_available()` checked at import time
- Used again at runtime to select benchmarking function
- CUDA streams detected and skipped (cause race conditions)

**File:** `BackendBench/eval.py` Lines 43-51, 164-166

### 7. Benchmarking
- GPU: `triton.testing.do_bench()` (handles synchronization)
- CPU: `cpu_bench()` (time.perf_counter fallback)
- Performance score: Geometric mean of speedups

**File:** `BackendBench/eval.py:eval_performance()` Lines 162-224

---

## Source Files Reference

All source code examined:

```
Core Execution:
├── src/config.py                                      (42 lines)
├── src/code_evaluator.py                            (222 lines)
├── src/utils.py                                     (151 lines)
└── src/modal_runner.py                              (105 lines)

GPU Kernels & Testing:
├── BackendBench/eval.py                             (283 lines)
└── BackendBench/utils.py                            (444 lines)
```

**Total Analyzed:** ~1,247 lines of source code

---

## Critical Code Locations

### GPU Configuration
- **File:** `src/config.py`
- **Lines:** 8-42
- **Key Class:** `BackendBenchConfig`
- **Key Parameter:** `gpu: str = "T4"`

### Local vs Remote Decision
- **File:** `src/code_evaluator.py`
- **Lines:** 113-140
- **Key Method:** `CodeEvaluator.__init__()`
- **Decision Logic:** `if self._gpu == "local"`

### Kernel Compilation
- **File:** `BackendBench/utils.py`
- **Lines:** 410-443
- **Key Function:** `compile_kernel_from_string()`
- **JIT Point:** `spec.loader.exec_module(module)`

### Test Execution
- **File:** `BackendBench/eval.py`
- **Lines:** 93-125 (correctness), 162-224 (performance)
- **Key Functions:** `eval_correctness_test()`, `eval_performance()`

### GPU Device Placement
- **File:** `BackendBench/utils.py`
- **Lines:** 100-113
- **Key Function:** `_deserialize_tensor()`
- **Default:** `device="cuda"`

### CUDA Stream Detection
- **File:** `BackendBench/utils.py`
- **Lines:** 50-97
- **Key Function:** `uses_cuda_stream()`
- **Reason:** Prevents benchmarking race conditions

---

## How to Use This Documentation

### Step 1: Quick Overview
Start with **GPU_EXECUTION_SUMMARY.txt** - read all sections for quick understanding

### Step 2: Find Specific Answers
Use **GPU_EXECUTION_QUICK_REFERENCE.md** - find answer to your question, read code example

### Step 3: Deep Dive
Read **GPU_EXECUTION_ANALYSIS.md** - full section for complete understanding with context

### Step 4: Code Exploration
Use file locations tables to jump to source code in your editor

---

## Key Insights

1. **Single Configuration Point**: GPU behavior controlled entirely by `gpu` parameter in `BackendBenchConfig`

2. **Clean Abstraction**: `CodeEvaluator.is_modal` property makes local/remote decision completely transparent

3. **Two Execution Paths**:
   - Local: Direct thread-pool execution
   - Remote: Modal RPC with full Docker container

4. **Triton Integration**: Automatic detection and JIT compilation via `@triton.jit` decorator

5. **GPU-Default Tests**: Test tensors created on GPU by default with CPU fallback

6. **Smart Benchmarking**: Automatic selection of `triton.testing.do_bench()` vs `cpu_bench()`

7. **CUDA Stream Safety**: Explicit detection and skipping of kernels using CUDA streams

---

## Testing the Knowledge

To verify your understanding, you should be able to answer:

1. What happens when `gpu="T4"` is set?
2. Where does Triton JIT compilation occur?
3. What is the default device for test tensors?
4. Why are CUDA streams detected and skipped?
5. What's the difference in test data handling between local and remote execution?
6. How does benchmarking work on GPU vs CPU?
7. Where is the local vs remote decision made?

All answers are in these documents with exact line numbers and code examples.

---

## Document Stats

| Document | Size | Sections | Tables | Code Examples |
|----------|------|----------|--------|---|
| GPU_EXECUTION_ANALYSIS.md | 8.5KB | 12 | 3 | 40+ |
| GPU_EXECUTION_QUICK_REFERENCE.md | 5.2KB | 7 | 2 | 25+ |
| GPU_EXECUTION_SUMMARY.txt | 6.8KB | 7 | 2 | 20+ |
| **Total** | **20.5KB** | **26** | **7** | **85+** |

---

## Notes

- All line numbers reference source files in `.venv/lib/python3.13/site-packages/`
- Code examples are exact excerpts from source, not paraphrased
- File paths are absolute for easy reference
- Generated by complete source code analysis, not from documentation

---

**Generated:** 2025-11-12  
**Analysis Method:** Complete source code exploration and extraction  
**Coverage:** 100% of GPU execution paths
