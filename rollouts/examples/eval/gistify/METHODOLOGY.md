# Gistify Eval - Methodology Reference

Based on: "Gistify! Codebase-Level Understanding via Runtime Execution"
arXiv: https://arxiv.org/abs/2510.26790

---

## 1. Task Definition (Section 2.1)

> "generate a single gistified file that reproduces the runtime behavior of the original codebase under the given command"

### Four Core Properties:

1. **Self-Contained**
   > "All necessary components from the given codebase must be included so that the gistified file can be executed standalone"

2. **Execution Fidelity**
   > "Executing the gistified file must replicate the original codebase's runtime behavior"

3. **Minimalism**
   > "Only the code essential to reproducing the runtime behavior should be preserved"

4. **Grounded Preservation**
   > "No hallucinated code may be introduced. All content must be derived directly from the original codebase"

---

## 2. System Prompt (Appendix A.4, Figure 2)

### Opening:
> "I've uploaded a python code repository in the directory {working dir}. There is an original test invocation (the command that reproduces behavior we want to preserve): problem statement Your job: create a single file named 'concise.py' saved at {working dir} that is **self-contained**, **minimal**, and **executable**, and when used in place of the original test run reproduces the same runtime behavior and outputs."

### Output Format:
> "Produce one file only: '{working dir}/concise.py'."
> "The assistant must return only the contents of 'concise.py' (no extra files, no analysis, no commentary)."

### HIGH-LEVEL RULES for creating 'concise.py':

#### Rule 1: Inline internal dependencies
> "Copy into 'concise.py' every function, class, or top-level code from the files inside {working dir} that is executed when running {problem statement}."
> "Do not use 'import' statements for modules defined in {working dir}."

#### Rule 2: Remove unexecuted lines
> "When copying lines in 'concise.py', keep only the lines that is actually executed when running {problem statement}."
> "Delete unused functions, classes, variables, if-else, imports, and unreachable branches."
> "Ensure the file remains syntactically correct and minimal after removal."

#### Rule 3: Preserve original source lines
> "Do not rewrite or reformat lines unless necessary to keep the files valid."
> "Do not arbitrary generate new lines that do not exist in the original {working dir} files."
> "You may adjust indentation, remove empty 'else' blocks, or adapt 'try-except' structures only when required to preserve correctness."

#### Rule 4: Keep external imports
> "Leave imports to external libraries, frameworks, or standard runtime libraries unchanged."
> "Only remove or inline dependencies that come from {working dir}."

#### Rule 5: No shortcuts or cheating
> "Do not stub, fake, or monkey-patch external modules."
> "Do not reimplement or newly add third-party libraries."
> "Do not hard-code outputs"
> "Do not replace test logic with simplified equivalents"

#### Rule 6: Preserve test behavior
> "The test function much remain unchanged, except for import adjustments needed to reference inlined code."
> "The output, exceptions, or exit codes must match the original run of {problem statement}."

#### Rule 7: Do not execute the code
> "Do not run or simulate the program (e.g., with 'pytest', 'python', or any other tools)"

---

## 3. Evaluation Metrics (Section 2.3)

### Metric 1: Execution Fidelity (Binary)

**Formula (Equation 1):**
> "𝟙[runs(c,𝒢) ∧ (out(c,𝒢) = out(c,𝒞))]"

**Definition:**
> "Execution Fidelity is a binary metric where 1 means the gistified file runs successfully and produces the same output"
> "tests pass/fail consistency and stdout/stderr matching"

**Evaluation Protocol:**
> "once the model generates the gistified file, to ensure that execution for evaluation is based on the original test, we integrate the test code from the original codebase to the gistified file and execute it."

### Metric 2: Line Execution Rate (Minimality)

**Formula (Equation 2):**
> "1/|L_exec(𝒢)| × Σ_{ℓ∈L_exec(𝒢)} 𝟙[ℓ is executed]"

**Definition:**
> "Line Execution Rate measures minimality by calculating the fraction of lines in the gistified file that are actually executed"

### Metric 3: Line Existence Rate (Grounding)

**Formula (Equation 3):**
> "1/Σ_{b∈ℬ_𝒢}|ℒ(b)| × Σ_{b∈ℬ_𝒢} Σ_{ℓ∈ℒ(b)} 𝟙{ℓ ∈ ℒ_𝒞(b)}"

**Definition:**
> "Line Existence Rate measures the proportion of code in the gistified file that is directly preserved from the original codebase"

**Block Matching (Appendix A.1):**
> "Lines outside of any block (e.g., top-level statements) are treated as standalone units"
> "block by block while respecting the code hierarchy"

**Normalization:**
- AST parsing to ignore formatting differences
- Remove comments
- Split multi-line imports
- Merge multi-line statements

---

## 4. Dataset Construction (Section 3.1, Appendix A.3)

### Repositories (Table 4):

| Repository | License | URL |
|------------|---------|-----|
| flask | BSD 3-Clause | github.com/pallets/flask |
| requests | Apache-2.0 | github.com/psf/requests |
| pylint | GPL 2.0 | github.com/pylint-dev/pylint |
| scikit-learn | BSD 3-Clause | github.com/scikit-learn/scikit-learn |
| seaborn | BSD 3-Clause | github.com/mwaskom/seaborn |
| debug-gym | MIT | github.com/microsoft/debug-gym |

### Test Selection:
> "we experiment with widely used GitHub repositories which are present in SWE-Bench"
> "We extract and filter test sets for each repository...remove tests whose execution is dependent on the test's file location"
> "For the main experiment, we evaluate over 25 tests for each of the 5 repositories"
> "we begin by extracting all available test cases, including parameterized ones"
> "we filter out environment-dependent tests, such as those requiring relative file paths or fixed module locations"

### Parameterized Tests:
> Grouped by base structure; all parameter instances evaluated together

---

## 5. Experimental Setup (Section 3.1)

### Models:
> "Our evaluation spans four leading LLM variants: GPT-5, GPT-5-mini, Claude-3.7-Sonnet, and Claude-Sonnet-4"

### Limits:
> "We use a 128K token limit for all models"
> "All experiments run are capped at 50 steps"

### Frameworks:
> "We conduct experiments using three widely adopted open-sourced frameworks: SWE-Agent...GitHub Copilot...Mini-SWE-Agent"

### Execution Tools:
> "for the agentic models, we exclude the execution tools ('python', 'pytest') in the default setting where execution is disabled"

### Environment:
> "run them in the same Docker environment, using the current version of the repositories"

---

## 6. Implementation Notes

### What We Need to Build:

1. **Test Case Sampler**
   - Clone repos at consistent commits
   - Extract pytest commands from test files
   - Filter environment-dependent tests

2. **GistifyEnvironment**
   - Provide codebase access (read-only or with tools)
   - Format the system prompt with {working_dir} and {problem_statement}
   - Capture model output (concise.py)

3. **Scoring Pipeline**
   - **Execution Fidelity**: Run original test, run gistified test, compare outputs
   - **Line Execution Rate**: Use Python trace/coverage to measure executed lines
   - **Line Existence Rate**: AST-based matching between gistified and original code

4. **Test Integration**
   - After model generates concise.py, inject original test function
   - This prevents cheating by modifying test logic

### Open Questions:

1. Exact commit hashes for repos? (Paper says "current version")
2. How to handle test fixtures and conftest.py?
3. Timeout for test execution?
4. How to normalize output comparison (timestamps, memory addresses, etc.)?
