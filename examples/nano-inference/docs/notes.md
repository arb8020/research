what makes inference slow
decoding is sequential
suppose you have 3 users, each want 100 tokens
generating per-user sequentially - not using the batch dim!

