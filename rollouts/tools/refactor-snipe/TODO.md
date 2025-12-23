# refactor-snipe TODO

## Language Support

Currently only Python is supported. To add more languages:

1. Install tree-sitter grammar: `pip install tree-sitter-{lang}`
2. Add to `treesitter.py` LANGUAGES and EXTENSION_TO_LANG dicts
3. Create code generator in `langs/{lang}.py`
4. Add language-specific scope queries to `scope.py` (or make it configurable per-language)

### Priority Languages to Add
- [ ] JavaScript/TypeScript (port from refactoring.nvim `treesitter/langs/typescript.lua`)
- [ ] Go (port from refactoring.nvim `treesitter/langs/go.lua`)
- [ ] Rust
- [ ] C/C++
- [ ] Java

Reference configs are in `/tmp/refactoring.nvim/lua/refactoring/treesitter/langs/`

## Refactoring Operations

### Implemented
- [x] Extract Function (`extract-function`) - extract code region into new function
- [x] Inline Variable (`inline-variable`) - replace variable uses with its value
- [x] Extract Variable (`extract-variable`) - extract expression to variable
- [x] Inline Function (`inline-function`) - replace function calls with body

## Future Ideas
- [ ] Rename Symbol (across file)
- [ ] Extract Constant
- [ ] Move to File (extract function to new file)
- [ ] Add/Remove Parameter
