# csearch TODO

## Performance Optimizations

### Parallelization
- [ ] Multiprocessing for `build_index` - parallelize file reading + trigram extraction
- [ ] Multiprocessing for tree-sitter parsing in `search_definitions` / `search_references`
- [ ] Batch SQLite inserts during indexing (fewer transactions)

### Index Size (currently 1GB for kubernetes)
- [ ] Only index identifiers, not full file content
- [ ] Compress posting lists
- [ ] Consider bloom filters for approximate matching
- [ ] Incremental index updates (don't rebuild from scratch)

### Search Quality
- [ ] Fuzzy/substring matching for symbol names
- [ ] Ranking results by relevance
- [ ] Support regex patterns in queries

## Features

### Semantic Search
- [ ] Embeddings backend (sentence-transformers + FAISS)
- [ ] Hybrid search: trigram candidates → embedding rerank

### Language Support
- [ ] Add more tree-sitter grammars (C, C++, Java, etc.)
- [ ] Better ctags fallback for unsupported languages

### Usability
- [ ] JSON output format (`--json`)
- [ ] Watch mode for index (`csearch index --watch`)
- [ ] Config file (`.csearchrc`) for default options
- [ ] Respect `.gitignore`

## Benchmarks

kubernetes (28k files, 16k indexable):
- Index build: 4 min (single-threaded)
- Index size: 1GB
- Search without index: 16s
- Search with index: 1.3s
- ripgrep equivalent: 0.2s
