team has been having lots of fun adventures lately around multi-turn tokenization

one of our key design goals is to ensure that environments are portable across models, including as evals for API models which might not expose tokenizers, and robustly supporting chat semantics while "hiding" all of the tokenizer footguns that come up during RL training

"just do token-in-token-out" always is sound advice for standalone research efforts, but then your tokenizer creeps up into your environment, and then it can no longer be a general-purpose eval

since june or july, verifiers+prime-rl has always only ever *trained* on tokens+logprobs which directly come out of the model

however, there are some cases where this can still result in off-policyness in terms of *input* retokenization

we support two major rollout strategies for token sequences -- "branching" + "interleaved" -- and handle this a bit differently for each

the branching case is the easiest -- each turn is a separate prompt-completion sample, you can get both the true input+output tokens from vLLM for free, and then store this directly for training. exactly mirrors deployed usage. this is basically necessary for models which truncate their reasoning, or harnesses which rewrite context.

however, it's also pretty inefficient, as it doesn't allow you to overlap repeated prefixes across turns. ideally, you want something that behaves like prefix caching, but at the trainer level.

addressing this inefficiency is the goal of the interleaved thinking strategy, designed to allow overlapping prefixes both for instruct models and interleaved-thinking models

previously, we did a form of "surgery" in constructing the training sequences where we splice the output tokens in between the deterministically-tokenized prompt/environment tokens, but this has its limits -- if using chat inputs, a model's previous response may be retokenized differently, and so the output tokens are trained against a prompt which differs in tokenization from what was seen at inference.

we'd been living with this for a while, runs were stable at the scale we'd been training, but mismatch metrics were still higher than desired in multi-turn rollouts + we knew we needed a better approach

now, thanks to 
@mikasenghaas
 we've fixed this by properly accounting for token-in-token-out at the full sequence level for interleaved rollouts without previous-output retokenization, yet still with simple chat semantics exposed to the user

eventually, i expect we'll want a single strategy that gives the best of both worlds, interleaving as much as we can via TITO, but branching where we have to for rewritten prompt histories

but we'll leave that for another day :)

---

https://github.com/PrimeIntellect-ai/prime-rl/pull/1422

---

Most RL frameworks are fundamentally unstable.

We wasted more H100 hours on debugging this than any other issue fornour multi-turn, multi-env RL run (below).

When using OpenAI-style messages for env interactions, parsing and retokenizing leads to subtly different tokens. This creates extremely unlikely tokens, which dominate the gradient and over time lead to collapse. The screenshots describe the mechanism in more detail.

We tried a lot of interventions, but ended up reimplementing our environments to use token lists directly (Tokens-in/Tokens-out). This fixed it immediately.

Always inspect logprobs!

---

Training
SID-1 was trained using Magistral’s modified version of GRPO without SFT. In the sections below, we chronicle a few important algorithmic and infrastructure observations discovered in development.

Tokens-In/Tokens-Out
Many reinforcement learning frameworks use OpenAI-style messages for their environment interaction. This has many benefits, such as being able to use the inference engines’ inbuilt tool call parsers and compatibility with API-models for SFT data generation.

Messages are practicable for offline RL, single-turn environments, or settings which use few chat template features (no thinking, no tool calling, for example). For multi-turn environments with many tool calls, using the messages abstraction invariably leads to model collapse. Parsing a token list to a message list is lossy. For example, it erases whitespace information around tool calls. Applying the chat template to generate the next turn then subtly changes the tokens. When inspecting logprobs, we observe extremely unlikely tokens where these shifts occurred (see example in Figure 4).

Generation
Training
query
":
"
\"
A
Love
That
query
":
" \"
A
Love
That
Log probability of token:
-20
0
Figure 4: Log probabilities of tokens for a trained 14B model in a non-TI/TO respecting pipeline. At rollout time, the byte sequence " \" is generated as two tokens, which gets transformed into one extremely low probability token after retokenization.

Training on these tokens leads to instability and collapse over time. We trace this collapse to feedback loops:

At generation time, the model produces two categories of rollouts, 
G
G(ood) and 
B
B(ad) rollouts, for example 
B
B rollouts could have incorrectly formatted tool calls.
Passing the bad rollouts from the generation engine to the training engine causes them to appear like good rollouts: 
apply_chat_template
(
parse
(
B
)
)
=
G
′
apply_chat_template(parse(B))=G 
′
 .
At training time, the model sees 
(
G
′
)
(G 
′
 ) with negative advantage, whereby some tokens in 
G
′
G 
′
  have extremely negative logprobs and thus dominate the gradient, driving these logprobs even more negative and the model to generate fewer good rollouts in general.
We find this explains the gradual performance increase followed by catastrophic collapse patterns observed in prior work, which we depict in Figure 5.Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (Jin et al., 2025) We find that simply ensuring that all rollouts are processed by our pipeline in a strictly Tokens-In/Tokens-Out (TI/TO) manner is sufficient to prevent these extreme training instabilities without the use of importance sampling ratios.Your Efficient RL Framework Secretly Brings You Off-Policy RL Training (Yao et al., 2025)

Figure 5: Training on malformed tokens sees reward increase initially. After some time, we observe a decrease in tool calling accuracy follwed by fully degenerate outputs. This collapses the reward and is not recoverable.
Figure 5: Training on malformed tokens sees reward increase initially. After some time, we observe a decrease in tool calling accuracy follwed by fully degenerate outputs. This collapses the reward and is not recoverable.

Length Normalization
Dr. GRPOUnderstanding R1-Zero-Like Training: A Critical Perspective (Liu et al., 2025) observes that per-sequence length normalization incentivizes the model to find shorter positive advantage and longer negative advantage rollouts. To alleviate this "length bias" they normalize per group instead. MagistralMagistral (Mistral AI, 2025) and most RL frameworks follow this approach.

<b>Figure 6:</b> Length distribution for positive and negative advantage rollouts (1848 rollouts, model-generated tokens only). Negative advantage rollouts are generally longer.
Figure 6: Length distribution for positive and negative advantage rollouts (1848 rollouts, model-generated tokens only). Negative advantage rollouts are generally longer.

We find experimentally that removing the length bias leads to unreliability over long training runs. Specifically, we observe the model starts producing out-of-vocabulary tokens. We posit the following root cause: By removing the length bias, the mean per token advantage is no longer guaranteed to be zero. As negative advantage rollouts tend to be longer, the mean per token advantage is negative, which we depict in Figure 6. This suppresses the logits of allocated tokens globally; over time, as assigned logit values collapse, the model begins to sample allocated tokens from the tokenizer's vocabulary (which are included as embedding matrix dimensions are rounded to have a high power of 2 factors). This regression appears robust to hyperparameters, and we prove that the gradients flowing to the logits of these unallocated tokens will be positive if the advantage has this length bias removed.

Let 
o
i
,
t
o 
i,t
​
  be the 
t
t-th token of the 
i
i-th rollout and 
L
i
=
∣
o
i
∣
L 
i
​
 =∣o 
i
​
 ∣ be the length of rollout 
i
i. We can formulate both the standard and length-debiased GRPO in the same objective formula by defining the following per-token advantages 
A
~
i
A
~
  
i
​
  from the raw advantages 
A
i
A 
i
​
 :

Length-biased:

A
~
i
=
A
i
L
i
⋅
G
A
~
  
i
​
 = 
L 
i
​
 ⋅G
A 
i
​
 
​
 
Unbiased:

A
~
i
=
A
i
∑
j
L
j
A
~
  
i
​
 = 
∑ 
j
​
 L 
j
​
 
A 
i
​
 
​
 
Either of these can be put into the objective formula below to achieve the GRPO surrogate policy objectives:

J
(
θ
)
=
∑
i
=
0
G
∑
t
=
0
L
i
J
i
,
t
(
θ
)
where 
J
i
,
t
(
θ
)
=
min
⁡
(
r
θ
(
i
,
t
)
A
~
i
,
clip
(
r
θ
(
i
,
t
)
,
1
−
ϵ
,
1
+
ϵ
)
A
~
i
)
and 
r
θ
(
i
,
t
)
=
π
θ
(
o
i
,
t
∣
o
i
,
<
t
)
π
o
l
d
(
o
i
,
t
∣
o
i
,
<
t
)
.
J(θ)
​
  
= 
i=0
∑
G
​
  
t=0
∑
L 
i
​
 
​
 J 
i,t
​
 (θ)
where J 
i,t
​
 (θ)=min(r 
θ
(i,t)
​
  
A
~
  
i
​
 ,clip(r 
θ
(i,t)
​
 ,1−ϵ,1+ϵ) 
A
~
  
i
​
 )
and r 
θ
(i,t)
​
 = 
π 
old
​
 (o 
i,t
​
 ∣o 
i,<t
​
 )
π 
θ
​
 (o 
i,t
​
 ∣o 
i,<t
​
 )
​
 .
​
 
Lemma 1: For length-debiased GRPO, the 
correlation
correlation between the sequence length and the raw advantage 
correlation
(
A
i
,
L
i
)
correlation(A 
i
​
 ,L 
i
​
 ) is proportionate to the sum of the per-token advantages 
∑
i
=
0
G
∑
t
=
0
L
i
A
~
i
∑ 
i=0
G
​
 ∑ 
t=0
L 
i
​
 
​
  
A
~
  
i
​
 , in particular they share the same sign.

Proof. We can calculate:

correlation
(
A
i
,
L
i
)
∝
covariance
(
A
i
,
L
i
)
=
1
G
∑
i
=
0
G
L
i
A
i
−
(
1
G
∑
i
=
0
G
L
i
)
(
1
G
∑
i
=
0
G
A
i
)
  
=
0
=
1
G
∑
i
=
0
G
∑
t
=
0
L
i
A
i
∝
∑
i
=
0
G
∑
t
=
0
L
i
A
~
i
.
correlation(A 
i
​
 ,L 
i
​
 )
​
  
∝covariance(A 
i
​
 ,L 
i
​
 )
= 
G
1
​
  
i=0
∑
G
​
 L 
i
​
 A 
i
​
 − 
( 
G
1
​
  
i=0
∑
G
​
 L 
i
​
 )( 
G
1
​
  
i=0
∑
G
​
 A 
i
​
 )
​
  
=0
 
= 
G
1
​
  
i=0
∑
G
​
  
t=0
∑
L 
i
​
 
​
 A 
i
​
 
∝ 
i=0
∑
G
​
  
t=0
∑
L 
i
​
 
​
  
A
~
  
i
​
 .
​
 
This shows that if there is a negative 
correlation
correlation between the lengths of rollouts and their advantages, we should expect for the per-token advantage term to be negative. Intuitively, we can see that this would depress 
r
θ
(
i
,
t
)
r 
θ
(i,t)
​
  and thus increase the probability of rare tokens, and we will see that this can be made into a precise statement.

□
□

Lemma 2: In expectation, 
A
~
i
A
~
  
i
​
  has a negative linear relationship to the gradients of logits of unsampled (out-of-vocabulary) tokens with respect to the GRPO objective under necessary assumptions.

Proof. We will treat the case of being exactly on-policy, i.e. 
π
o
l
d
=
π
θ
π 
old
​
 =π 
θ
​
  so that the derivatives of the per-token loss can be treated as the derivatives of a log-policy-likelihood:

d
J
i
,
t
=
A
~
i
  
d
(
log
⁡
π
θ
(
o
i
,
t
)
)
dJ 
i,t
​
 = 
A
~
  
i
​
 d(logπ 
θ
​
 (o 
i,t
​
 ))
Suppose that the probability distribution for a specific token 
π
θ
(
o
i
,
t
)
π 
θ
​
 (o 
i,t
​
 ) is defined by the softmax of some vector of logits 
l
(
i
,
t
)
l 
(i,t)
  and that the logit corresponding to an out-of-vocabulary token (not equal to 
o
i
,
t
o 
i,t
​
 ) is 
l
o
o
v
(
i
,
t
)
l 
oov
(i,t)
​
 . Calculating the derivative of the sample objective with respect to this logit gives:

∂
J
∂
l
o
o
v
(
i
,
t
)
=
∂
J
i
,
t
∂
l
o
o
v
(
i
,
t
)
=
A
~
i
∂
∂
l
o
o
v
(
i
,
t
)
log_softmax
(
l
(
i
,
t
)
)
o
i
,
t
=
−
A
~
i
e
l
o
o
v
(
i
,
t
)
∑
o
e
l
o
(
i
,
t
)
∂l 
oov
(i,t)
​
 
∂J
​
 = 
∂l 
oov
(i,t)
​
 
∂J 
i,t
​
 
​
 = 
A
~
  
i
​
  
∂l 
oov
(i,t)
​
 
∂
​
 log_softmax(l 
(i,t)
 ) 
o 
i,t
​
 
​
 =− 
A
~
  
i
​
  
∑ 
o
​
 e 
l 
o
(i,t)
​
 
 
e 
l 
oov
(i,t)
​
 
 
​
 
Crucially, the derivative of log-softmax is always positive and independent of the selected token 
o
i
,
t
o 
i,t
​
 , as long as this token is not the out-of-vocabulary token itself. Taking expectations over this token, conditioned on it not being out-of-vocabulary:

E
(
∂
J
∂
l
o
o
v
(
i
,
t
)
|
o
i
,
t
≠
oov
)
=
−
E
(
A
~
i
|
o
i
,
t
≠
oov
)
e
l
o
o
v
(
i
,
t
)
∑
o
e
l
o
(
i
,
t
)
E( 
∂l 
oov
(i,t)
​
 
∂J
​
  
​
 o 
i,t
​
 ≠oov)=−E( 
A
~
  
i
​
  
​
 o 
i,t
​
 ≠oov) 
∑ 
o
​
 e 
l 
o
(i,t)
​
 
 
e 
l 
oov
(i,t)
​
 
 
​
 
□
□

Together, these lemmas draw a clear path from negative 
correlation
correlation between the lengths of rollouts and their advantages, to 
A
~
i
A
~
  
i
​
  being on average negative, to the logits of unsampled tokens being increased when 
J
J is maximised with backpropagation.

Length Scheduling. We find that compute is optimally allocated throughout the training run by starting at a low maximum length for rollouts which is gradually increased over time. We also leverage a soft length penaltyMagistral (Mistral AI, 2025),CWM: An Open-Weights LLM for Research on Code Generation with World Models (FAIR CodeGen team, 2025) to enable retrieval and length-based training signals to be propagated from the same rollout.

Parallel tool use and hierarchical retrieval. In Figure 7 we show that our model learns to use multiple tools in parallel without further intervention. This reduces end-to-end latency by reducing the number of round trips to the retrieval server. We also introduce hierarchical retrieval, loosely inspired by the web search tools in gpt-ossgpt-oss-120b & gpt-oss-20b Model Card (OpenAI, 2025) and OpenPipe art-e.ART·E: How We Built an Email Research Agent That Beats o3 (Corbitt, 2025) The initial search only provides short excerpts from the documents. If the model wants to read the full content, it can selectively "read" the document by using a read tool. This architecture reduces token usage and allows the model to see more documents before exhausting context window limits. For many questions, this lets SID-1 use fewer input tokens than reranking.

<b>Figure 7:</b> The turn count stays constant while the number of tool calls increases, showing that our model learns to make multiple search requests per turn. More tool calls per turn are desirable, as it decreases round trips to the retrieval backend.
Figure 7: The turn count stays constant while the number of tool calls increases, showing that our model learns to make multiple search requests per turn. More tool calls per turn are desirable, as it decreases round trips to the retrieval backend.




---

ah, so to add some more color to this: we have always been doing token-out. in the beginning, we had to hack this into vllm, but its been officially supported for some time now. so we always get the decoded tokens + their logprobs straight from the engine.  

now, we have been holding back on token-in because a) vllm + vf don't support token-in natively (yet) and b) if your rl algorithm is designed for off-policy (which it is anyways these days bc we wanna train async + have trainer-inference mismatch etc) the edge cases of retokenization issues that remain just becomes another instance of "off-policy" tokens and simply get masked. we have been training in this regime just fine

that said, it does seem like the natural thing to do to just move fully into token land. here's the problem i found: basically, if you are always in token land, you may not respect chat template formatting because chat templates are inherently in text land. consider this basic 2 turn example with token in/out:
- turn 1: prompt u1, and response a1
- turn 2: prompt u1,a1,u2, where u1 and a1 are just the concatenated raw token sequences from turn 1. u2 is tokenized with the chat template and concatenated as well
the qwen3 chat template puts a newline in between chat messages. so if you just gave u1,a1,u2 as messages to the tokenizer, you'd have a newline in between each message. in our case, we are missing a newline between u1,a1 and u2 because a1 ends on the end message token + u2 was tokenized independently of other messages so it doesn't have a leading newline. i know this issue is subtle and one could ofc hardcode a fix for this specific case, but it points at the more fundamental issue that you may train ood w.r.t to random formatting your chat template does which means you train ood w.r.t to 99% of inference deployments which are text-in/text-out oai spec

am i paranoid because of one newline? maybe, but i have seen chat template stuff fucking up stuff too many times to not be scared. if you can tell me im wrong or that this just isnt an issue, id be more than happy because then yesterday's work wasn't for nothing lol
