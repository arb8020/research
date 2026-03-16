# Terminal Bench Task-Level Data Summary

**Data Collection Date:** March 10, 2026
**Sources:** tbench.ai, arXiv:2601.11868, marginlab.ai

## Executive Summary

Terminal Bench 2.0 contains **89 tasks** across 10 technical domains. The benchmark paper reports that **frontier models and agents score less than 65%** overall, with top performer GPT-5.2 + Codex CLI achieving **62.9% ± 3.0%**.

### Key Findings

1. **No per-task pass rates are publicly available** - Only aggregate model-level performance is reported
2. **Difficulty distribution** (from registry analysis of 247 tasks including variants):
   - Easy: ~8 tasks
   - Medium: ~41 tasks
   - Hard: ~31 tasks
3. **Task completion times** (from Table 1 in paper):
   - Expert: 48.6% complete in <1hr, 47.3% take 1-24hrs
   - Junior: 8.1% complete in <1hr, 71.6% take 1-24hrs

## Overall Model Performance (Terminal Bench 2.0)

### Leaderboard Top 10 (as of March 2026)

| Rank | Agent | Model | Pass Rate | Stderr |
|------|-------|-------|-----------|--------|
| 1 | Forge Code | Gemini 3.1 Pro | 78.4% | ±1.8% |
| 2 | Droid | GPT-5.3-Codex | 77.3% | ±2.2% |
| 3 | Simple Codex | GPT-5.3-Codex | 75.1% | ±2.4% |
| 4 | Terminus-KIRA | Gemini 3.1 Pro | 74.8% | ±2.6% |
| 5 | Terminus-KIRA | Claude Opus 4.6 | 74.7% | ±2.6% |
| 6 | Mux | GPT-5.3-Codex | 74.6% | ±2.5% |
| 7 | OB-1 | Multiple | 72.4% | ±2.3% |
| 8 | TongAgents | Claude Opus 4.6 | 71.9% | ±2.7% |
| 9 | Junie CLI | Multiple | 71.0% | ±2.9% |
| 10 | CodeBrain-1 | GPT-5.3-Codex | 70.3% | ±2.6% |

**Claude Code** (Opus 4.6): Rank 33 with **58.0%** accuracy

### Terminal Bench 1.0 Leaderboard Top 10

| Rank | Agent | Model | Pass Rate | Stderr |
|------|-------|-------|-----------|--------|
| 1 | Apex2 | claude-4-5-sonnet | 64.5% | ±1.1% |
| 2 | Chaterm | claude-4-5-sonnet | 63.7% | ±1.1% |
| 3 | Abacus AI Desktop | Multiple | 62.3% | ±1.8% |
| 4 | Ante | claude-sonnet-4-5 | 60.3% | ±2.1% |
| 5 | Droid | claude-opus-4-1 | 58.8% | ±1.7% |
| 6 | Droid | claude-sonnet-4-5 | 57.5% | ±1.5% |
| 7 | OB-1 | Multiple | 56.7% | ±1.2% |
| 8 | Ante | claude-sonnet-4 | 54.8% | ±2.9% |
| 9 | Droid | gpt-5 | 52.5% | ±4.1% |
| 10 | Chaterm | claude-sonnet-4-5 | 52.5% | ±1.0% |

## Task Inventory (154 Confirmed Tasks from Registry)

### Sample Tasks by Difficulty

#### HARD Tasks (Confirmed Samples)
- **blind-maze-explorer-5x5** - Systematically explore unknown 5x5 maze, build map
- **pytorch-model-cli** - Build C CLI tool for PyTorch MNIST inference
- **sam-cell-seg** - Cell segmentation using Segment Anything Model
- **make-mips-interpreter** - Create MIPS architecture interpreter
- **play-zork** - Speedrun text adventure game
- **protein-assembly** - Bioinformatics protein structure assembly
- **parallelize-graph** - Graph algorithm parallelization
- **cartpole-rl-training** - Reinforcement learning task
- **bn-fit-modify** - Bayesian Network DAG recovery and intervention
- **cancel-async-tasks** - Async task runner with cleanup handling
- **circuit-fibsqrt** - Logic gate circuit for fib(isqrt(N))

#### MEDIUM Tasks (Sample)
- adaptive-rejection-sampler
- analyze-access-logs
- blind-maze-explorer-algorithm
- build-linux-kernel-qemu
- chess-best-move
- conda-env-conflict-resolution
- cron-broken-network
- db-wal-recovery
- fibonacci-server
- git-workflow-hack
- npm-conflict-resolution
- pcap-to-netflow
- pytorch-model-recovery
- vim-terminal-task
- ode-solver-rk4
- configure-git-webserver
- polyglot-c-py
- swe-bench-astropy-1
- tmux-advanced-workflow
- caffe-cifar-10
- llm-spec-decoding
- hf-lora-adapter
- spinning-up-rl
- gcc-compiler-optimization
- torch-pipeline-parallelism
- mcmc-sampling-stan
- weighted-max-sat-solver
- matlab-python-conversion

#### EASY Tasks (Confirmed Samples)
- **csv-to-parquet** - Convert CSV file to Parquet format
- hello-world
- fix-permissions
- create-bucket
- extract-safely
- grid-pattern-transform
- simple-web-scraper
- processing-pipeline

### Task Domains (from Registry)

1. **Software Engineering** - Git workflows, debugging, package management
2. **System Administration** - QEMU, kernel compilation, server configuration
3. **Security** - Cryptanalysis, vulnerability exploitation, certificate management
4. **Machine Learning** - Model training, inference, RL, PyTorch/TensorFlow
5. **Data Science** - Data processing, visualization, statistical analysis
6. **Scientific Computing** - Numerical methods, simulation, R/MATLAB
7. **Games** - Maze solving, chess analysis, CoreWars
8. **Build Tools** - Compilation, dependency resolution, cross-compilation
9. **Networking** - Protocol analysis, packet processing, server setup
10. **File Operations** - Format conversion, compression, parsing

## Complete Task List (Alphabetical)

1. accelerate-maximal-square
2. acl-permissions-inheritance
3. adaptive-rejection-sampler
4. add-benchmark-lm-eval-harness
5. aimo-airline-departures
6. analyze-access-logs
7. audio-synth-stft-peaks
8. bank-trans-filter
9. blind-maze-explorer-5x5
10. blind-maze-explorer-algorithm
11. bn-fit-modify
12. break-filter-js-from-html
13. build-cython-ext
14. build-initramfs-qemu
15. build-linux-kernel-qemu
16. build-pov-ray
17. build-tcc-qemu
18. c-to-safe-rust
19. caffe-cifar-10
20. cancel-async-tasks
21. cartpole-rl-training
22. catch-me-if-you-can
23. causal-inference-r
24. chem-property-targeting
25. chess-best-move
26. code-from-image
27. compile-run-compcert
28. conda-env-conflict-resolution
29. configure-git-webserver
30. count-call-stack
31. count-dataset-tokens
32. crack-7z-hash
33. cron-broken-network
34. cross-entropy-method
35. csv-to-parquet
36. custom-memory-heap-crash
37. db-wal-recovery
38. debug-long-program
39. decommissioning-service-with-sensitive-data
40. enemy-grid-escape
41. extract-moves-from-video
42. feal-differential-cryptanalysis
43. feal-linear-cryptanalysis
44. filter-js-from-html
45. financial-document-processor
46. find-official-code
47. fix-code-vulnerability
48. fix-ocaml-gc
49. fix-pandas-version
50. flood-monitoring-basic
51. fmri-encoding-r
52. gcode-to-text
53. gcc-compiler-optimization
54. get-bitcoin-nodes
55. git-leak-recovery
56. git-workflow-hack
57. grid-pattern-transform
58. hf-lora-adapter
59. hf-model-inference
60. hf-train-lora-adapter
61. home-server-https
62. html-finance-verify
63. huarong-dao-solver
64. hydra-debug-slurm-mode
65. implement-eigenvectors-from-eigenvalues-research-paper
66. incompatible-python-fasttext
67. install-klee-minimal
68. install-windows-3
69. install-windows-xp
70. interactive-maze-game
71. jq-data-processing
72. jupyter-notebook-server
73. kv-store-grpc
74. large-scale-text-editing
75. llm-inference-batching-scheduler
76. llm-spec-decoding
77. log-summary-date-ranges
78. logistic-regression-divergence
79. make-doom-for-mips
80. make-mips-interpreter
81. matlab-python-conversion
82. mcmc-sampling-stan
83. merge-diff-arc-agi-task
84. mixed-integer-programming
85. mixed-integer-programming-1
86. mnist-learning-fix
87. model-extraction-relu-logits
88. modernize-fortran-build
89. modernize-scientific-stack
90. multi-source-data-merger
91. multistep-definite-integral
92. neuron-to-jaxley-conversion
93. new-encrypt-command
94. nginx-request-logging
95. npm-conflict-resolution
96. ode-solver-rk4
97. openssl-selfsigned-cert
98. organization-json-generator
99. pandas-sql-query
100. parallel-particle-simulator
101. parallelize-compute-squares
102. parallelize-graph
103. path-tracing-reverse
104. pcap-to-netflow
105. play-zork-easy
106. polyglot-c-py
107. polyglot-rust-c
108. postgres-csv-clean
109. predicate-pushdown-bench
110. predict-customer-churn
111. protocol-analysis-rs
112. prove-plus-comm
113. pytorch-model-cli
114. pytorch-model-recovery
115. qemu-alpine-ssh
116. rare-mineral-allocation
117. recover-accuracy-log
118. recover-obfuscated-files
119. reshard-c4-data
120. rstan-to-pystan
121. run-pdp11-code
122. sam-cell-seg
123. schemelike-metacircular-eval
124. security-celery-redis-rce
125. security-vulhub-minio
126. setup-custom-dev-env
127. simple-sheets-put
128. simple-web-scraper
129. slurm-simple-node-monitoring
130. solve-maze-challenge
131. sparql-professors-universities
132. speech-to-text
133. spinning-up-rl
134. spring-messaging-vul
135. sql-injection-attack
136. sqlite-db-truncate
137. sqlite-with-gcov
138. stable-parallel-kmeans
139. sudo-llvm-ir
140. super-benchmark-upet
141. swe-bench-astropy-1
142. swe-bench-astropy-2
143. swe-bench-fsspec
144. swe-bench-langcodes
145. tmux-advanced-workflow
146. torch-pipeline-parallelism
147. torch-tensor-parallelism
148. train-bpe-tokenizer
149. tree-directory-parser
150. unprivileged-headless-pyrender
151. vim-terminal-task
152. vimscript-vim-quine
153. weighted-max-sat-solver
154. winning-avg-corewars

## Data Limitations

### What Is NOT Available

1. **Per-task pass rates** - Individual task success rates by model/agent are not published
2. **Per-task difficulty for all tasks** - Only sampled difficulties confirmed via individual task pages
3. **Which specific models solved which tasks** - Aggregate data only
4. **Historical task performance trends** - Only current leaderboard snapshots

### What IS Available

1. **Overall model/agent pass rates** - Aggregate success across all 89 tasks
2. **Difficulty labels** - Easy/Medium/Hard classifications exist but not systematically extracted
3. **Task descriptions** - Available on individual task pages at tbench.ai
4. **Task categories and tags** - Domains, technologies, skill areas
5. **Human completion time estimates** - Expert vs junior developer benchmarks

## Paper Key Insights (arXiv:2601.11868)

### Performance Alignment
- **93.3% of human-hard tasks are also empirically hard** for models
- Empirical difficulty determined by frontier model pass rates

### Task Composition (Table 1)
**Expert Completion Times:**
- 36 tasks (48.6%): < 1 hour
- 35 tasks (47.3%): 1-24 hours

**Junior Completion Times:**
- 6 tasks (8.1%): < 1 hour
- 53 tasks (71.6%): 1-24 hours

### Model-Agent Results (from Paper Table 2)
| Model | Agent | Resolution Rate |
|-------|-------|-----------------|
| GPT-5.2 | Codex CLI | 62.9% ± 3.0% |
| Claude Opus 4.5 | Terminus 2 | 57.8% ± 2.5% |
| GPT-OSS-20B | - | 3.1% ± 1.5% |

## Degradation Tracking Data

### Claude Code (Opus 4.6) Recent Performance
- **Current (Mar 10, 2026):** 56% (50 test cases)
- **7-day average:** 53% (250 test cases)
- **30-day average:** 56% (1,450 test cases)
- **Baseline:** 56%

### Codex (gpt-5.4-xhigh) Recent Performance
- **Current (Mar 10, 2026):** 57% (28/49 test cases)
- **7-day average:** 55% (250 test cases)
- **30-day average:** 55% (250 test cases)

## Methodology Notes

This data was compiled from:
1. **Public leaderboards** at tbench.ai (TB 1.0 and 2.0)
2. **Registry pages** showing 247 task variants
3. **Paper abstract and HTML version** (full PDF not accessible)
4. **Degradation trackers** for Claude Code and Codex
5. **Individual task pages** for difficulty confirmation

**Limitation:** Per-task pass rates appear to be proprietary or not yet published. The benchmark focuses on aggregate model-level evaluation rather than task-level difficulty analysis.

## Raw Data Sources

- Terminal Bench 2.0 Leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.0
- Terminal Bench 1.0 Leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/1.0
- Task Registry (head): https://www.tbench.ai/registry/terminal-bench-core/head
- Paper: https://arxiv.org/abs/2601.11868
- Task Explorer: https://www.tbench.ai/tasks
- Claude Code Tracker: https://marginlab.ai/trackers/claude-code
- Codex Tracker: https://marginlab.ai/trackers/codex

## Next Steps for Obtaining Per-Task Data

To get actual per-task pass rates, you would need to:

1. **Contact the authors** - The marginlab/laude-institute team may have this data
2. **Run the benchmark yourself** - Install terminal-bench and evaluate models
3. **Check paper appendices** - Download full PDF and examine figures 11-13 (mentioned but not extracted)
4. **Access raw results** - If leaderboard submissions include task-level breakdowns in their reports
5. **Monitor future releases** - TB 3.0 and TB-Science 1.0 are in progress

## Conclusion

While Terminal Bench provides comprehensive task diversity (89 tasks, 10 domains, multiple difficulty levels), **per-task success rate data is not publicly available**. Only aggregate model-level performance is reported on leaderboards and in the paper. The benchmark appears designed for holistic agent evaluation rather than granular task-level analysis.

The data shows a clear performance hierarchy:
- **Top tier (75-78%):** Gemini 3.1 Pro, GPT-5.3-Codex with advanced agents
- **Mid tier (55-70%):** Claude Opus 4.6, various agent implementations
- **Low tier (<10%):** Open-weight models, basic baselines

This suggests the benchmark successfully differentiates model capabilities while maintaining sufficient difficulty to avoid saturation at the high end.
