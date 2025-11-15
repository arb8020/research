# Handoff: Frontend UI Improvements

## Current Task
Implementing 5 UI enhancements for the rollouts frontend. Commit each one separately as you complete them.

## What's Left to Do

### 1. Model Dropdown (IN PROGRESS)
**Status**: Backend done, frontend needs completion

**Backend**: Added `GET /api/models` endpoint in `server.py` that fetches from OpenAI and Anthropic APIs (no fallback hardcoded models)

**Frontend TODO**:
- Add `loadModels()` function to fetch from `/api/models`
- Populate dropdown with `{id, provider, name}` from response
- Call from `init()` function
- Group by provider in UI (optional but nice)

### 2. GPU ID Selection
**Currently**: Single number input
**Need**: Checkboxes/toggles for GPU ranks 0-7 (multi-GPU selection)
- Allow selecting multiple GPUs
- Save as array in config

### 3. Sample Range from Dataset
**Currently**: Manual number inputs
**Need**: Auto-populate from actual dataset size
- When dataset loads, read its length
- Set max value for range slider
- Show "0 to {dataset_length}"

### 4. Dataset Path File Browser
**Currently**: Text input
**Need**: Mini file explorer for selecting dataset files
- Browse `datasets/` directory
- Click to select .json/.jsonl files
- Backend endpoint to list files in directory

### 5. Max Turns -1 Support
**Currently**: Just number input
**Need**: Support `-1` for unlimited turns
- Add checkbox or special input for "unlimited"
- Validate and save `-1` when unlimited selected

## Key Files

### Backend
- `rollouts/frontend/server.py` - API endpoints
  - Line 71: Added `/api/models` route
  - Line 547: Added `_list_models()` method

### Frontend
- `rollouts/frontend/index.html` - Single file app
  - Line 702: Model dropdown (needs JS to populate)
  - Line 1461: `init()` function - add `loadModels()` call here
  - Line 743: GPU ID field (replace with checkboxes)
  - Line 746: Dataset path field (replace with browser)
  - Line 751: Sample range (auto-populate from dataset)
  - Line 776: Max turns (add -1 unlimited support)

## Code Style
- See `~/research/docs/code_style/` for Tiger Style guidelines
- Explicit state, no magic
- Comments explain WHY not WHAT
- Vanilla JS, no frameworks

## Commit Pattern
Commit each feature separately with message format:
```
Add dynamic model loading from OpenAI/Anthropic APIs

- Backend: GET /api/models fetches from both providers
- Frontend: loadModels() populates dropdown on init
- No hardcoded fallback models

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Testing
Start server: `cd ~/research/rollouts && python -m rollouts.frontend.server --project ~/wafer_stuff/kernels-gpumode-agent`

User has API keys in env for testing model fetching.

## Notes
- User wants changes one at a time with commits
- No fallback hardcoded values - must fetch from APIs/datasets
- Keep three-pane layout intact
- Message builder integration was just completed (works well)
