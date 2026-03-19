# Prime CI Coverage

This directory mirrors the PrimeIntellect nightly task surface, but with local
honesty about what is and is not actually supported in this repository.

Current coverage policy:

- real configs where we have a local pathway worth exercising
- explicit stubs where Prime has coverage but we do not yet have a trustworthy
  local implementation

Status is recorded per file via `config_status`.

Current real configs:

- `reverse_text/eval_api.py`
- `reverse_text/rl_sync.py`
- `reverse_text/rl_async.py`
- `reverse_text/rl_megatron.py`
- `alphabet_sort/rl.py`

Current stubs:

- `wordle/rl.py`
- `wiki_search/rl.py`
- `hendrycks_sanity/rl.py`
- `acereason_math/rl.py`
- `multimodal_color_codeword/rl.py`
