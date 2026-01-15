[ ] language failed returns as green in the UI
[ ] activity -> click on feedback -> should take me to the appropriate sample + turn (only takes me to main eval page rn)
[ ] check if we store metadata about provider/etc fails
[ ] dump code/commit hash so that its easy to know exactly what ran (check config fingerprinting?)

---

## Evaluation Traces & Dashboard Flow Analysis

### Overview

The evaluation system follows this flow:
1. **Eval runs** → Generates `report.json` + `samples/*.json` files locally
2. **Upload** → `scripts/upload_traces.py` or `base_config.py` auto-upload pushes to Supabase Storage
3. **Dashboard** → `trace-viewer` module fetches from Supabase and renders traces

### Data Structures

#### 1. Report File (`{run_name}/report.json`)
Top-level metadata for an evaluation run:
```json
{
  "eval_name": "gpt52_smoke_00",
  "dataset_path": "gpt52_smoke_00",
  "total_samples": 4,
  "summary_metrics": {
    "mean_correct": 0.5,
    "mean_compiled": 0.5,
    "mean_speedup": 0.42,
    "mean_fast_1": 0.25,
    "success_rate": 1.0,
    "avg_turns": 1.0,
    "avg_tokens": 4221.0
  },
  "config": {
    "endpoint": { "provider": "openai", "model": "gpt-5.2-...", ... },
    "max_samples": 4,
    "max_concurrent": 4,
    "evaluation_timestamp": "..."
  },
  "timestamp": "2025-12-23T01:11:17.216723",
  "sample_ids": ["sample_0001", "sample_0000", ...]
}
```

#### 2. Sample Files (`{run_name}/samples/{sample_id}.json`)
Two formats exist (dashboard normalizes both):

**Old format:**
```json
{
  "sample_id": "sample_0000",
  "input_data": { "level": "level1", "problem_id": 1, "name": "...", "ref_arch_src": "..." },
  "output_data": { "compiled": true, "correct": true, "speedup": 1.67 },
  "trajectory": { "messages": [...], "completions": [...] }
}
```

**New rollout format:**
```json
{
  "id": "sample_0000",
  "input": { "level": "level1", "problem_id": 1, ... },
  "trajectory": { "messages": [...], "completions": [...] },
  "score": { "metrics": [{ "name": "compiled", "value": 1 }, ...] },
  "metadata": { "sample_data": {...} }
}
```

#### 3. Feedback File (`{run_name}/feedback.json`)
User annotations stored alongside traces:
```json
{
  "version": 1,
  "feedback": [{
    "id": "uuid",
    "sample_id": "sample_0000",
    "turn_index": 2,
    "comment": "Bug in memory allocation",
    "selected_text": "cudaMalloc...",
    "line_start": 45,
    "category": "bug",
    "severity": "high",
    "author_id": "user-id",
    "author_email": "user@example.com",
    "created_at": "2025-01-03T...",
    "mentions": ["@alice"]
  }]
}
```

### Dashboard Loading Flow

1. **Run List View** (`useTraceLoader.loadFromCloud`):
   - Fetches `_manifest.json` → list of run names
   - Shows runs immediately, then lazy-loads reports in batches of 5
   - Reports are fetched from `{run_name}/report.json`

2. **Sample Table View** (`useTraceLoader.loadRun`):
   - Loads `report.json` + lists samples from bucket (more reliable than report.sample_ids)
   - Shows table immediately with sample IDs
   - Lazy-loads actual sample data in batches of 20

3. **Conversation View** (`ConversationView`):
   - Displays trajectory messages with expand/collapse
   - Shows feedback inline per turn
   - Supports text selection for adding feedback

### Key Files

| Component | Purpose |
|-----------|---------|
| `research/evals/kernelbench/base_config.py` | Eval configuration, runs evaluations, saves results |
| `research/evals/kernelbench/modal_app.py` | Modal GPU worker for kernel evaluation |
| `scripts/upload_traces.py` | CLI to upload results to Supabase |
| `apps/internal-dashboard/src/modules/trace-viewer/lib/trace-loader.ts` | Loads traces from cloud/local |
| `apps/internal-dashboard/src/modules/trace-viewer/lib/feedback-storage.ts` | Feedback CRUD operations |
| `apps/internal-dashboard/src/modules/trace-viewer/hooks/useTraceLoader.ts` | React hook for loading state |
| `apps/internal-dashboard/src/modules/trace-viewer/hooks/useFeedback.ts` | React hook for feedback state |

### Upload Flow

```
1. Eval completes → saves to research/evals/kernelbench/results/{run_name}/
   ├── report.json
   ├── samples/
   │   ├── sample_0000.json
   │   └── sample_0001.json
   └── trajectories/ (optional)

2. Upload (one of):
   a) Manual: python scripts/upload_traces.py results/{run_name}/
   b) Auto: config.output.upload_to_supabase = True → base_config.upload_results_to_supabase()

3. Supabase Storage (traces bucket):
   ├── _manifest.json          # { runs: ["run1", "run2", ...] }
   ├── {run_name}/
   │   ├── report.json
   │   ├── samples/*.json
   │   └── feedback.json       # Created when users add feedback
```

### TODOs & Improvement Ideas

- [ ] **Sample normalization**: `trace-loader.ts:normalizeSample()` handles format differences, but could be cleaner
- [ ] **Incremental sample loading**: Currently loads 20 at a time, could be smarter about priority (load visible first)
- [ ] **Viewed state persistence**: Uses `{run_name}/viewed.json` - consider IndexedDB for local-first approach
- [ ] **Feedback conflict resolution**: Multiple users editing simultaneously could overwrite - need optimistic locking
- [ ] **Large trace handling**: Some samples are 250KB+ - consider compression or chunking
- [ ] **Offline support**: Currently requires Supabase - could add local-first with sync
