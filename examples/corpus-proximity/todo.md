goal:

be able to write search(corpus, text) and get back nearest neighbors/distance/something


[x] basic data collect
[x] decided on fineweb-edu (nanochat uses this)
[x] single-threaded prepare_data.py
  [x] download N shards (fineweb-edu parquet files)
  [x] process shards (read parquet, chunk into paragraphs)
  [x] save as jsonl
  [x] print to verify it worked
[ ] parallelize with Worker pattern (fork + sockets)

[x] embed chunks (sentence-transformers or BERT or something)
[x] save embedded chunks as numpy arrays or something
[x] deploy.py for running on GPU instances

[ ] build a function to search a given query text and return top k

[ ] test by embedding sentence known in training data, get itself as nearest neighbor + vice versa (get a sentence we know is not in data)



