# Handoff: JSON Output Optimization for Outlier Analysis

**Date:** 2025-10-24
**Issue:** Result files are 400x larger than necessary (~2GB vs ~5MB)
**Impact:** 99.5% wasted storage, JSON parsing failures, memory issues

---

## 🐛 Problem Summary

The outlier analysis is storing **every single outlier occurrence** with full details `{layer, seq_pos, value}` in the final JSON output, even though only summary statistics are needed for classification.

### Current File Sizes

| Model | Systematic Features | Current Size | Wasted Space |
|-------|---------------------|--------------|--------------|
| OLMoE-1B-7B | 49 | 14MB | ~10MB (71%) |
| GPT-OSS-20B | 1,465 | 486MB | ~480MB (99%) |
| Qwen3-30B | 110 | 287MB | ~285MB (99%) |
| Mixtral-8x7B | 4,635 | 863MB | ~860MB (99.6%) |
| GPT-OSS-120B | 1,695 | 329MB | ~325MB (99%) |
| **Total** | | **~2GB** | **~1.96GB (98%)** |

### Observed Issues

1. **JSON parsing failures:** Large files (>300MB) fail to load with `json.load()` due to corruption/memory issues
2. **Unnecessary data:** The detailed occurrence lists are only used temporarily to compute summaries, then discarded
3. **Storage waste:** ~2GB of results could be ~5MB with same information

---

## 🔍 Root Cause

### File: `analyze_activations.py`

**Lines 118-127:** Store ALL occurrences
```python
for i in range(len(feature_indices)):
    feature_dim = feature_indices[i].item()
    seq_pos = seq_indices[i].item()
    value = values[i].item()

    feature_outliers[feature_dim].append({
        'layer': layer_name,      # ← Storing every occurrence
        'seq_pos': seq_pos,       # ← Storing every position
        'value': value            # ← Storing every value
    })
```

**Lines 175-189:** Compute summaries from stored occurrences
```python
for feature_dim, outlier_list in feature_outliers.items():
    # Extract summary stats from the detailed list
    layers_with_outlier = set(item['layer'] for item in outlier_list)
    layer_percentage = len(layers_with_outlier) / total_layers

    seq_positions_with_outlier = set(item['seq_pos'] for item in outlier_list)
    seq_percentage = len(seq_positions_with_outlier) / seq_len

    max_magnitude = max(abs(item['value']) for item in outlier_list)
```

**File: `run_full_analysis.py`, Line 222:** Store entire outlier_info in JSON
```python
batch_result = {
    ...
    "outlier_info": outlier_info,  # ← Includes all the detailed occurrences
    ...
}
```

### What Gets Stored

For **each feature dimension** → **each batch** → **each occurrence**:
- Example: Feature 292 in OLMoE has 1,869 stored occurrences
- Mixtral: 4,635 features × avg 100-1000 occurrences each = **millions** of dicts

---

## ✅ Proposed Solution

### Option 1: Incremental Summary (Recommended)

Compute summaries **while finding outliers** instead of storing all occurrences.

**Change in `find_outliers_in_activations()` (lines 118-127):**

```python
# BEFORE (stores all occurrences)
feature_outliers[feature_dim].append({
    'layer': layer_name,
    'seq_pos': seq_pos,
    'value': value
})

# AFTER (stores only summaries)
if feature_dim not in feature_outliers:
    feature_outliers[feature_dim] = {
        'layers': set(),
        'seq_positions': set(),
        'max_magnitude': 0.0,
        'count': 0
    }

stats = feature_outliers[feature_dim]
stats['layers'].add(layer_name)
stats['seq_positions'].add(seq_pos)
stats['max_magnitude'] = max(stats['max_magnitude'], abs(value))
stats['count'] += 1
```

**Change in `analyze_systematic_outliers()` (lines 175-189):**

```python
# BEFORE (iterates over occurrence list)
for feature_dim, outlier_list in feature_outliers.items():
    layers_with_outlier = set(item['layer'] for item in outlier_list)
    layer_percentage = len(layers_with_outlier) / total_layers

    seq_positions_with_outlier = set(item['seq_pos'] for item in outlier_list)
    seq_percentage = len(seq_positions_with_outlier) / seq_len

    max_magnitude = max(abs(item['value']) for item in outlier_list)

# AFTER (uses pre-computed summaries)
for feature_dim, stats in feature_outliers.items():
    layer_percentage = len(stats['layers']) / total_layers
    seq_percentage = len(stats['seq_positions']) / seq_len
    max_magnitude = stats['max_magnitude']
```

**Change return value (line 129-133):**

```python
# BEFORE
return {
    'feature_outliers': dict(feature_outliers),  # Contains full occurrence lists
    'layer_stats': layer_stats,
    'threshold': magnitude_threshold
}

# AFTER
return {
    'feature_outliers': {
        feature_dim: {
            'layers': list(stats['layers']),
            'seq_positions': list(stats['seq_positions']),
            'max_magnitude': stats['max_magnitude'],
            'count': stats['count']
        }
        for feature_dim, stats in feature_outliers.items()
    },
    'layer_stats': layer_stats,
    'threshold': magnitude_threshold
}
```

### Option 2: Don't Store outlier_info in Final Results

**Change in `run_full_analysis.py` (line 217-224):**

```python
# BEFORE
batch_result = {
    "batch_id": batch_idx + 1,
    "run_dir": str(run_dir),
    "sequences_processed": len(batch_texts),
    "systematic_outliers": systematic_outliers,
    "outlier_info": outlier_info,  # ← Remove this
    "timestamp": datetime.now().isoformat()
}

# AFTER
batch_result = {
    "batch_id": batch_idx + 1,
    "run_dir": str(run_dir),
    "sequences_processed": len(batch_texts),
    "systematic_outliers": systematic_outliers,
    # outlier_info removed - only systematic_outliers contains needed data
    "timestamp": datetime.now().isoformat()
}
```

---

## 📊 Expected Impact

### File Size Reduction

- **Before:** ~2GB for 5 models
- **After:** ~5-10MB for 5 models
- **Savings:** 99.5% reduction (400x smaller)

### Example: Mixtral-8x7B

- **Before:** 863MB
- **After:** ~2-3MB
- **What's kept:**
  - 4,635 feature summaries × ~500 bytes each = ~2.3MB
  - Layer stats, metadata, config = ~0.5MB
  - Total: ~3MB

### Benefits

1. ✅ **Fixes JSON parsing:** Files small enough to load with `json.load()`
2. ✅ **Same analysis results:** All classification data preserved
3. ✅ **Faster I/O:** 400x less data to read/write
4. ✅ **Less disk usage:** Save ~2GB per sweep
5. ✅ **Better debugging:** Smaller files easier to inspect

---

## 🧪 Testing Plan

1. **Implement Option 1** (incremental summary) in `analyze_activations.py`
2. **Run small test:** OLMoE-1B-7B with 4 sequences
3. **Verify output:** Check that `final_analysis_results.json` contains same classification but smaller size
4. **Compare results:** Ensure systematic feature counts match previous runs
5. **Run full sweep:** If test passes, rerun all 7 models

### Validation Checks

```python
# Verify same systematic features identified
old_features = set(f['feature_dim'] for f in old_results['all_systematic_outliers'])
new_features = set(f['feature_dim'] for f in new_results['all_systematic_outliers'])
assert old_features == new_features, "Feature sets must match"

# Verify file size reduction
old_size_mb = old_file.stat().st_size / (1024**2)
new_size_mb = new_file.stat().st_size / (1024**2)
assert new_size_mb < old_size_mb / 100, "Should be <1% of original size"
```

---

## 📁 Files to Modify

1. **`analyze_activations.py`**
   - Function: `find_outliers_in_activations()` (lines 118-133)
   - Function: `analyze_systematic_outliers()` (lines 175-189)

2. **`run_full_analysis.py`** (Optional - if using Option 2)
   - Batch result creation (line 217-224)

3. **Tests to update** (if any exist)
   - Any tests checking `outlier_info` structure

---

## 🔄 Migration Notes

### Backward Compatibility

- **Breaking change:** Yes - changes `outlier_info` structure
- **Impact:** Existing result files still readable, but won't match new format
- **Mitigation:** This is fine - old results are archive, new format is better

### Re-running Analysis

**Do we need to re-run?**
- Current 5 completed models: **Yes, if you want smaller files**
- Classification results won't change
- Can keep current results for now, just optimize going forward

**Priority:**
1. Fix code first ✅
2. Run remaining 2 models (Qwen3-Next-80B, GLM-4.5-Air) with new format ✅
3. Optionally re-run 5 completed models for consistency ⚠️

---

## 📝 Related Issues

- JSON files >300MB fail to parse completely
- Memory issues when loading large result files
- Slow result syncing (2GB over network vs 5MB)

---

## 🎯 Next Steps

1. [ ] Review this handoff with team
2. [ ] Decide: Option 1 (incremental) vs Option 2 (remove outlier_info) vs both
3. [ ] Implement changes in `analyze_activations.py`
4. [ ] Test with small model (OLMoE, 4 sequences)
5. [ ] Verify classification results match
6. [ ] Deploy fix and run remaining 2 models
7. [ ] Consider re-running completed models for consistency

---

**Questions?** Check the code in:
- `analyze_activations.py:118-133` (where occurrences are stored)
- `analyze_activations.py:175-189` (where summaries are computed)
- `run_full_analysis.py:222` (where outlier_info is saved)
