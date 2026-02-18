# Jane Street Archaeology Puzzle

## Puzzle
- Model at `jane-street/2025-03-10` (file: `model_3_11.pt`)
- Takes two-word text input like "vegetable dog", outputs a number
- Hint: "Maybe start by looking at the last two layers"
- Goal: figure out what the model does

## Model Architecture
- Type: `torch.nn.Sequential`
- 5442 layers (2721 Linear + 2721 ReLU alternating)
- 288,998,553 parameters (~1.16 GB)
- Input: 55 dimensions
- Output: 1 dimension

## Key Finding: Stack Overflow
- `model(x)` segfaults due to 5442 recursive calls in Sequential.forward()
- Fix: manual loop `for layer in model.children(): x = layer(x)`

## Last Two Layers Analysis

### Last Linear (48→1)
```
weight: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
bias: -15
```
Pattern: 16 ones, 16 negative-twos, 16 ones

### Second-to-Last Linear (192→48)
- 48 outputs = 3 groups of 16
- Each row reads 8 consecutive inputs with weights `[1,2,4,8,16,32,64,128]` → binary to int conversion
- Groups differ by exactly 1 in biases (second derivative check)
- Input 192 = 24 bytes × 8 bits

### Bias Values (Group 2 - middle group)
```
[-199, -239, -101, -35, -60, -64, -170, -50, -194, -185, -172, -227, -117, -149, -250, -124]
```

## Interpretation
The last layers implement a **character matching circuit**:
1. 192 inputs represent 24 characters in binary (8 bits each)
2. Weights `[1,2,4,8,16,32,64,128]` convert bits → byte values
3. Biases are negative target ASCII values
4. The `[1,-2,1]` pattern across groups checks second derivative (equality)
5. Output is 0 if and only if input matches target characters

## First Layer (55→224)
- Mostly identity: copies inputs 0-54 to outputs 0-54
- 165 rows with exactly 1 nonzero weight (value 1.0)
- 59 rows all zeros
- Input encoding: 55 dimensions = ?

## HuggingFace Repo Files
- `model.pt` - original, has custom forward with string preprocessing, Python <3.11 only
- `model_3_11.pt` - regenerated for Python 3.11+, lost the custom preprocessing
- The README just says "use model_3_11.pt for Python 3.11+"

## Open Questions
- How is the string "vegetable dog" encoded into 55 dimensions?
- Original model.pt had custom forward that accepted strings, but we can't load it on 3.13
- What are the 24 target characters the last layers check for?

## First Layer Analysis (55 → 224)
- Each input i maps to outputs [i, i+56, i+112] (3 copies)
- 59 outputs are all zeros (indices 55, 111, 167-223)
- All weights are 0 or 1, all biases are 0
- Creates 3 identical copies of the input for later comparison

## Hypotheses for Input Encoding (55 dims)
Most likely: `55 = 26 + 26 + 3` = letter counts for word1 + letter counts for word2 + 3 extras
- input[0-25] = count of 'a'-'z' in word1
- input[26-51] = count of 'a'-'z' in word2
- input[52-54] = ??? (lengths? checksums?)

## Testing Results
- Using letter-count encoding (26+26+3 dims), all tested inputs give pre-ReLU = -15.0, post-ReLU = 0.0
- The -15.0 is the bias of the final layer, meaning the weighted sum is always 0
- Anagram pairs produce identical layer 5438/5439 outputs:
  - "dog cat" == "god tac"
  - "cat dog" == "act god"
- This confirms the model uses letter counts, treating anagrams identically

## Layer 5438 (Comparison Layer) Behavior
- Pre-ReLU values range from -200 to +140
- Post-ReLU has ~24 of 48 values nonzero
- The 3 groups of 16 each differ by exactly 1 (second derivative pattern)
- The biases encode target values: [-200, -240, -102, -36, -61, -65, -171, -51, ...]

## Model Structure Summary
- 5442 layers total (2721 Linear + 2721 ReLU)
- ~63 repeating blocks of ~84 layers each
- 125 unique weight patterns, 62 appear multiple times
- Many weights are identical across blocks (systematic structure)

## Key Insight: Second Derivative Check
The final layers compute:
```
output = ReLU(sum(group1) - 2*sum(group2) + sum(group3) - 15)
```
This is a **discrete second derivative** formula: f''(x) ≈ f(x-1) - 2f(x) + f(x+1)

The groups differ by exactly 1 in biases, so this checks if the 16 comparison values follow a linear pattern.

## Open Questions
1. What is the model checking for? (anagram? rhyme? some word relationship?)
2. Why 63 blocks? (7*9? max word lengths?)
3. What input produces output > 0?

## Puzzle Submission
The puzzle asks to "figure out what it does" - the answer is likely a description of the computation, not a specific input.

Email: archaeology@janestreet.com
