goal: 

be able to write search(corpus, text) and get back nearest neighbors/distance/something


[ ] basic data collect
[x] decided on fineweb-edu (nanochat uses this)
[ ] single-threaded prepare_data.py
  [ ] download N shards (fineweb-edu parquet files)
  [ ] process shards (read parquet, chunk into paragraphs)
  [ ] save as jsonl
  [ ] print to verify it worked
[ ] parallelize with Worker pattern (fork + sockets)

[ ] embed chunks (sentence-transformers or BERT or something)
[ ] save embedded chunks as numpy arrays or something

[ ] build a function to search a given query text and return top k

[ ] test by embedding sentence known in training data, get itself as nearest neighbor + vice versa (get a sentence we know is not in data)



