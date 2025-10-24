Why Muon?
0:00
This video is about
0:00
how to find good parameters for machine learning model.
0:04
The search for good parameters is known as optimization,
0:07
and the tool we use is known as an optimizer.
0:10
For a long time, the Adam atomizer has been the default choice.
0:14
But, now there's a new, exciting challenger, the Muon optimizer.
0:18
The Muon optimizer is getting increasingly more attention
0:21
in the machine learning community.
0:23
It's delivering impressive results on small language models.
0:26
And it's about twice as computationally efficient as AdamW.
0:31
In other words, you can train faster, use less memory,
0:34
and still get great results.
Reviewing Adam
0:36
Let's first revisit Adam.
0:38
In standard supervised learning, we have a model
0:41
that makes predictions based on the input data.
0:45
At first, these predictions are just random guesses
0:47
since the model's parameters are initialized randomly,
0:51
We use the training data to compute the gradient of the loss
0:54
with respect to each parameter.
0:57
The gradient acts like a guide showing us which direction
1:00
the parameters should move to reduce the loss most effectively.
1:04
By updating the parameters in this direction, at each step,
1:08
the model gradually becomes better at making accurate predictions.
1:13
This is known as gradient descent.
1:16
Adam, builds on gradient descent, but maintains
1:18
two exponential moving averages of variables,
1:22
one for the past gradients themself called momentum,
1:26
and another for the square gradients.
1:28
Rather than updating parameters directly from the raw gradients,
1:32
Adam combines the momentum with an adaptive scaling
1:35
vector derived from the squared gradients.
1:38
This allows Adam to converge more quickly
1:41
and often and achieve better results than standard gradient descent.
1:45
But there's a catch.
1:47
Adam requires keeping two extra variables for every model parameter.
1:52
As a result, the optimizer states
1:54
takes up about twice as much memory as the model itself.
1:59
Furthermore, Adam treats all the parameters
2:01
as a single long vector, updating each value independently
2:06
without considering any internal structure.
2:09
This approach is called a vector-based optimizer.
Linear layer
2:13
But can we explicitly account for the underlying
2:16
matrix structure of the model parameters.
2:20
Linear layers are especially common in neural networks.
2:24
Here a linear layer transforms the input vector
2:27
x into an output vector z.
2:30
Each output z is just a weighted sum of the input x,
2:35
where the weights themselves are the trainable parameters.
2:39
We can describe this input output relationship
2:41
concisely using a matrix-vector product.
2:45
To update the weights, we first calculate
2:48
the momentum for each parameter.
2:51
But with vector based advisors like Adam,
2:54
the momentum for a linear layer, naturally, a 2D matrix
2:58
tends to become almost low-rank in practice.
3:02
This means that only a small number of dominant directions really drive
3:06
the updates.
3:07
Why? Many other directions contribute very little.
3:11
The Muon Optimizer tackles this issue
3:13
by orthogonalizing the momentum matrix.
3:16
By doing so, Muon amplifies the effect of rare directions.
3:21
the directions that typically receive small or infrequent updates.
3:25
Even though these rare the directions seems minor,
3:28
they are often essential for effective learning and can help
3:32
capture a more nuanced pattern in the data.
3:35
Let's get more concrete.
3:37
Suppose we have a 2D momentum matrix which we call M.
3:41
Orthogonalization is the process of finding a new matrix O
3:45
that is as close as possible to M, but with the special property
3:50
that its rows and columns are all also orthogonal to each other.
3:54
A key property of orthogonal matrix
3:57
is is that their transpose is also their inverse.
4:01
To build some intuition, imagine a momentum matrix as a single point
4:05
in the high dimensional space of all possible matrices.
4:09
Our objective is to find the nearest matrix O
4:12
to M that satisfies the orthogonality condition.
4:16
This sounds hard!
4:18
Luckily we have a powerful method for this
4:21
Singular Value Decomposition.
Solving orthogonalization with SVD
4:24
Let's make this more intuitive with a concrete example.
4:28
A 2D matrix M defines a linear transformation.
4:32
Think of applying the matrix M to the standard basis vectors.
4:36
When we multiply m by [1, 0], we get [2, 0].
4:40
This is just the first column of the matrix M.
4:44
Next, if we multiply M by [0, 1]
4:47
we get [1.5, 1.0], which is the second column.
4:53
In a sense a 2D matrix records how the transformation moves
4:57
the basis vectors to new position in space.
5:01
These four numbers fully determine how we transform
5:04
any input 2D vectors.
5:06
Remarkably, any
5:08
linear transformation can be broken down into three steps:
5:11
a rotation followed by a stretching or shrinking along each axis,
5:16
and then another rotation.
5:19
Mathematically, this process is called
5:21
Singular Value Decomposition or SVD.
5:25
It allows us to express any 2D matrix as the product
5:29
of three special matrices U, S and V transpose.
5:34
When we apply a linear transformation to a vector,
5:37
we can think of it as first rotating by V transpose,
5:40
then scaling each coordinate by the diagonal entry in S,
5:44
and finally rotating again by new.
5:48
Both U and V are orthonormal matrices,
5:51
which means that their rows and columns are mutually orthogonal
5:55
and have unit lengths.
5:57
This means that we can use SVD to tackle the orthogonalization
6:01
problem.
6:02
By computing the SVD of our momentum matrix,
6:05
then setting all the singular values in S to one,
6:09
we obtain the orthogonal matrix we want.
6:12
Easy, right?
6:14
But, performing
6:15
SVD on the matrix is computationally intensive.
6:19
We cannot afford running this step
6:21
for every update iteration when training our model.
6:25
Luckily, there is an efficient alternative.
Newton-Schulz iteration - Odd polynomial matrix
6:28
We can use something called an odd polynomial matrix.
6:32
This function takes a matrix X as input,
6:35
and computes the weighted sum of X
6:38
and X X transpose times X.
6:42
But why might this be useful?
6:44
Let's unpack how this helps with our 2D matrix example.
6:48
First, we can rewrite the equation by factoring out the matrix M.
6:53
Then we substitute for M using its SVD form.
6:57
Notice that the product between the V and V transpose equals
7:01
the identity matrix since V is orthonormal.
7:06
Because S is a diagonal matrix, multiplying it
7:09
by itself, just square each of its diagonal entries.
7:13
As we distributed the matrix multiplications, certain
7:16
terms like U transpose U simplify to the identity,
7:20
leading to a much cleaner expression.
7:23
In the end, we see that the left side of the equation has the matrix
7:27
U, while the right side has the matrix V transpose.
7:32
As a result, we can
7:33
combine the terms to further simplify the expression.
7:37
This says that applying an odd polynomial matrix function
7:40
to M acts on is singular values, in the same way
7:44
as applying the function to each singular value individually,
7:48
then reconstructing the matrix with the original singular vectors.
7:53
This principle applies to any odd polynomial,
7:56
including higher order variants like a fifth-order polynomial.
8:00
By choosing appropriate values for the coefficient a, b, and c,
8:05
we can push the singular values closer to one, all
8:08
without explicitly computing the SVD.
Newton-Schulz iteration - Example
8:12
Suppose we set a to 1.5,
8:14
b to -0.5 and c to 0,
8:19
we can visualize the effect of this function on a singular value
8:22
by plotting its input output relationship.
8:26
The red curve represents how an input values
8:28
x is mapped to the output value. Y.
8:32
We will focus on the input range between 0 and 1
8:35
since our singular values will fall within this interval.
8:39
Our goal here is to turn any input values
8:42
to an output value that is closer to one.
8:45
To visualize this, we plot the yellow dots
8:48
evenly spaced along the x axis between 0 and 1.
8:53
After applying the function, notice
8:55
how these yellow dots are moved toward one on the y axis.
8:59
This is great.
9:01
Let's see what happens when we apply this function repeatedly.
9:05
Each time we apply the function, all points are closer to one.
9:10
In the plots below, we can see how the functions
9:13
behaves over several iterations.
9:16
After five iterations, almost all input
9:18
values end up very close to one.
9:21
Now let's try changing the coefficients
9:23
to see how this affects the transformation.
9:27
For example, we might say a equals to two
9:29
be equal to -1.5 and c equals to 0.5.
9:35
here.
9:35
The red curve shows the new odd polynomial function.
9:39
By plotting the function after each iteration,
9:42
we can see that the values converge to one even more quickly
9:46
with these coefficients.
9:48
But is this the best we can do?
9:51
Let's tune the value of a, b, c so that we can get even faster.
9:55
convergence.
9:57
First, increasing the value of a is crucial
10:01
as this coefficient primarily control how quickly
10:04
small initial singular values converge towards one.
10:08
Second, it turns out empirically we don't have
10:11
to make the singular values converge to one exactly.
10:15
We just need them to be bounded by a certain range,
10:18
for example, between 0.7 and 1.3.
10:23
This leads to a two coefficients of a, b and c.
10:28
after just a few iterations, we can map any singular values
10:31
between 0 and 1 to the desired range.
The Muon optimizer
10:35
With this trick, we can now write down the algorithm.
10:39
For each update iteration, we first compute the gradient GT
10:43
using backpropagation.
10:45
Then, update the momentum
10:47
as an exponential moving average of the past gradients.
10:51
Next we normalize the 2D momentum matrix
10:54
so that it has unit norm.
10:56
This ensures that the initial singular values
10:58
are all between 0 and 1.
11:01
We repeat this also
11:02
orthogonalization process five times to get matrix O.
11:06
And then use O to update the parameters.
11:09
Each iteration involves only matrix multiplications,
11:13
which it can be efficiently computed with GPUs
11:16
without the need to compute SVD.
11:19
This method is called MomentUm Orthogonalization
11:23
by Newton-Schultz or Muon.
11:26
But when scaling up to train a larger model,
11:29
the performance gains over AdamW diminish.
11:33
To resolve this issue, we add a weight
11:35
decay mechanism as used in AdamW.
11:39
In addition, we adjust the learning rate
11:41
by taking account the size of the 2D matrix.
11:45
The two improvements help stabilize the training of large models.
The exploding attention logit crisis
11:49
But, there's still a challenge.
11:51
Researchers have observed that as training continues,
11:55
the attention logits can grow larger and larger,
11:58
which may cause the training process to become unstable.
12:02
Where does that come from and how can we fix it?
12:06
Consider a simple scenario.
12:08
Suppose we have a sequence of four tokens.
12:12
Each token is mapped to a embedding vector of dimension d.
12:16
Let's call the matrix of these embeddings X.
12:20
For simplicity, we will focus on self-attention
12:23
in the first transformer block
12:24
although the situation is similar in all layers.
12:28
We obtain the query key and value representations by projecting
12:33
the input embedding x with the parameter matrix WQ, WK and WV.
12:39
Next, the attention mechanism computes a weighted sum of the value
12:43
vectors where the weights are determined by the attention scores.
12:49
The attention logics before the softmax are computed by
12:52
multiplying the query matrix Q with the transpose of the key metrics K.
12:57
Here we substitute the expressions of Q and K into the formula.
13:02
And further simplifying the attention calculation.
13:06
Note that the matrix X and its transpose denotes
13:09
the embedding vectors,
13:10
which are typically normalized to have unit norms.
13:14
To prevent the attention logics from becoming excessively large,
13:18
we must control the scale of W_Q and W_K.
13:22
A common strategy is to apply a scaling vector to these matrixes.
13:27
During training we monitor the maximum value of the attention
13:30
logics.
13:32
If it exceeds a certain threshold tau,
13:34
we calculate a scaling ratio, denoted as gamma.
13:38
The idea is simple.
13:40
When the attention logics surpass the threshold tau,
13:43
we simply scales the relevant model parameters
13:46
by the factor gamma to keep them in check.
13:50
Because both W_Q and W_K contribute to the attention logics,
13:55
we scale each of these metrics by the square root of gamma.
13:59
The revised algorithm looks like this.
14:02
First we update the
14:03
model parameters theta using Muon optimizer.
14:07
Next, if the maximum attention logics is larger than tau,
14:11
we rescale both W_Q and W_K
14:14
by multiplying them with the square root of gamma.
14:18
This trick is called QK-clip.
14:21
By doing so, we directly constrain the attention logics, ensuring
14:25
that they stay within a safe range
14:27
by rescaling the query and key projection weights.
14:31
This looks great.
14:33
But, in practice self-attention consists of multiple heads.
14:37
We achieve this by splitting the query,
14:39
key, and value matrices into several heads.
14:43
Four in our example.
14:45
For each hand we regroup Q, K, V and we compute attention
14:50
independently and concatenate outputs from all heads
14:54
and project them with an output matrix W_O.
14:59
When the maximum attention logics go beyond the threshold,
15:02
it does not make sense to rescale all the heads in the same way.
15:07
Instead, we introduce an individual scaling
15:09
factor for each head to control their logics separately.
MuonClip: Extending QK-clip to Multi-head Latent Attention (MLA)
15:14
But things get tricky if we want to use Multi-head
15:17
latent attention (MLA) proposed by DeepSeek.
15:21
MLA compresses the query key and value representations
15:25
into a low-rank space to reduce the size of the KV cache.
15:30
This compression is performed using a down projection matrix,
15:34
which produce latent representations.
15:37
These compressed latent vectors are then map
15:39
back to the query key and values for each attention.
15:43
Hence, using the corresponding up-projection matrixes.
15:47
But, a challenge arise because this low-rank key value
15:51
compression does not work with rotary position embedding.
15:55
To overcome this limitation, researchers propose a decoupled RoPE
15:59
technique which introduces extra multi-head queries
16:03
and a shear key to encode positional information.
16:07
For MLA, the query
16:09
key and values are regrouped for each head.
16:13
Specifically for the query, we concatenate the compressed
16:16
query component Q^C with the rotated query Q^R.
16:22
Similarly, the key is constructed by concatenating the compressed
16:26
key K^C with its rotated counterpart K^R.
16:32
Using mult-head latent attention,
16:34
we need to carefully decide how to rescale these four matrices.
16:39
For the up-projection matrices.
16:40
we rescale the parameters for each head individually.
16:44
The RoPE component deserves special attention.
16:48
In this setup, each head has its own rotary query W^{QR},
16:52
but all hands share a single rotary key matrix W^{KR}.
16:58
If we were to apply the same per-head scaling for both,
17:02
the shared W^{KR} matrix will be rescaled
17:05
multiple times, which is undesirable.
17:08
To handle this properly, we rescale only the has specific
17:12
rotary queries W^{QR} by their respective gamma_h.
17:17
where leaving the shared rotary key matrix W^{CR} unchanged.
17:22
This technique is called MuonClip.
Results of MuonClip
17:24
Let's compare training with and without MuonClip.
17:28
with MuonClip applied as shown on the right.
17:32
The maximum attention logics are effectively capped
17:35
and quickly stabilized,
17:37
demonstrating the effectiveness of QK-clip regulation.
17:41
This helps the
17:42
optimizer maintain steady and reliable training.
17:46
I hope you enjoyed this overview.
17:48
Thanks for watching and I'll see you next time.
