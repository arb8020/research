# Cut the Bill, Keep the Turns: Affordable Multi-Turn Search RL

**Jiahao Wu**, **Zhongwen Xu**, **Qiang Fu**, and **Wei Yang**

*Tencent · TEG · AIPD*

*December 2025*

---

## TL;DR

- Using the [**rLLM](https://github.com/rllm-org/rllm) [[1](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] training framework + Qwen3-8B [[2](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] base model + offline Wikipedia / BrowseComp-Plus [[3](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] corpora + the synthetic multi-turn data from ASearcher [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]**, you can cost-effectively train a search agent that reliably performs 10+ retrieval turns.
- **SFT is not strictly necessary**: starting from the base model and doing GRPO-based [[5](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] RL is enough to build stable multi-turn search capability. SFT can provide a better starting point for long-horizon training, but it tends to lock the model into a particular pattern and makes it overuse multi-turn search even for simple questions.
- **Multi-turn search training dataset [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] is critical**: standard single-hop / two-hop datasets (HotpotQA [[6](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], etc.) do not train long-horizon multi-turn search; you need a dedicated synthetic multi-turn dataset.
- **Training stability is mostly an engineering problem**: the key issues are train/inference mismatch, strict token-in-token-out alignment, and handling “abnormal trajectories”.
- **Summarizing retrieved documents with an auxiliary LLM [[7](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]** improves training stability, supports longer horizons, and generalizes to setups that directly return raw search results.

---

## 1. The Research Bottleneck for Multi-Turn Search Agents

Recently, LLM **reasoning** and **tool-use** capabilities have improved rapidly: on the one hand, the open-source community has released many strong models [[2](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[8](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]; on the other hand, when moving to **multi-turn, long-horizon search-and-verify** settings (multiple retrievals, iterative disambiguation, evidence aggregation, and converging to a final answer), there is still a clear gap, where **smaller open-source models often lag far behind closed-source commercial models in both effective use of search turns and final accuracy**.

![image.png](attachment:74ed3d7b-8e86-477d-ac36-0a1de25a0d84:image.png)

To narrow this gap, the community has proposed many effective approaches [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[7](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[9](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[10](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[11](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[12](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]. Most follow a pipeline of **synthetic data construction → SFT alignment → RL optimization** to gradually push smaller models toward more robust long-horizon search strategies. However, for individual researchers, it is still hard to use these methods as a starting point: many are **not fully open-sourced**, key training details are **not fully disclosed**, and training stability is **highly sensitive to engineering details** (data synthesis and cleaning, tool environment stability, concurrency and timeouts, truncation and abnormal trajectory handling, train/inference alignment, etc.). In addition, **multi-stage training** is harder to reproduce, and **paid Web search APIs** further raise iteration cost.

This post provides a researcher-oriented tutorial for deploying and training a **multi-turn search agent baseline**. We systematize the key engineering choices and stability tricks we found in practice, so you can train your first multi-turn search agent at lower cost and with better reproducibility on classic multi-turn search benchmarks, **increasing average search turns from ~2 to 15+ and accuracy from single digits to ~30%**.

---

## 2. Training Cost

- **No complex multi-stage training**: a single-stage RL run is sufficient:

| Model | GPUs (train + retrieval + refine) | RL iters × bs | Time |
| --- | --- | --- | --- |
| Qwen3-8B | 16+8+8 | 300 × 128 | ~5 days |
| Qwen3-30B-A3B | 32+8+8 | 100 × 512 | ~5 days |
- **No expensive Web search APIs**: deploy a local retrieval server on offline corpora. To illustrate the savings, for this setup, using Serper API would cost about 1.5k USD for the RL stage alone:

> 300 RL iter × 1,024 rollouts / iter × 10 search call / rollout × 0.5 USD / 1,000 search call ≈ 1500 USD
> 

---

## 3. Data: Synthetic Multi-Turn Search Dataset

Standard single-hop / two-hop QA datasets (HotpotQA [[6](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], 2Wiki [[13](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], Musique [[14](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], etc.) can hardly train a multi-turn search agent. It is better to start from a dedicated synthetic multi-turn dataset. Many works [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[9](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[7](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[12](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] propose algorithms for constructing such datasets; we use the synthetic multi-turn dataset from **ASearcher [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]**, because its synthesis pipeline is largely based on Wikipedia and fits our offline (no Web search API) setup.

The raw dataset is noisy. We use a three-step filtering pipeline to improve the effective sample ratio and training stability:

1. **Remove Chinese samples**
    - Our wiki server does not handle Chinese retrieval well and may return garbled text;
    - There are about 2k Chinese-related samples in the ASearcher dataset, which we remove entirely.
2. **Remove math problems**
    - Use formula patterns / specific regexes to filter out math questions.
3. **Reject sampling with n=8 on the pretrained model**
    - Use an initial search agent (based on the pretrained base model) to roll out 8 trajectories per sample;
    - If all 8 trajectories have reward 0 or reward 1, discard the sample:
        - all 0s: too difficult / annotation issues / the environment fails to retrieve useful information → training signal is too weak;
        - all 1s: too easy / does not require multi-turn retrieval → wastes RL budget;

After filtering, we retain about **14k** usable samples. In practice, this step significantly increases the fraction of effective groups (equivalently increasing the effective batch size), which improves training stability.

---

## 4. Environment: Offline Retrieval Server

An offline retrieval server provides a stable environment, and trained agents still generalize reasonably well to web-like corpora. We use two environments:

- A local retrieval server based on **Wikipedia**, primarily used for training;
- A local retrieval server based on the **BrowseComp-Plus [[3](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] corpus**, used only for validation / testing.

BrowseComp [[15](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] is a widely used multi-turn web search agent benchmark, but evaluating on it typically requires a paid Web search API. To further reduce cost, we use BrowseComp-Plus [[3](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], which provides offline retrieval corpora aligned with BrowseComp, so we can evaluate by deploying a local retrieval server.

### 4.1 Wiki server

If you use the retrieval server from the rLLM examples as-is, it **cannot handle high-concurrency training workloads**. In practice, the following setup is more robust:

- **Build a GPU retrieval server based on FAISS-GPU + unicorn + Go**;
- Use a single 8 GPU machine as the wiki server, which yields:
    - total latency under 4 seconds for 1024 concurrent queries;
    - search success rate around **99%** during training (by tuning the timeout and the maximum active concurrency to balance throughput and stability).

### 4.2 BrowseComp-Plus server

The BrowseComp-Plus authors already provide the corpus and retrieval scripts. Our setup is straightforward:

- Start a retrieval server based on the BrowseComp-Plus corpus using the official implementation;
- **Use it only during validation / testing**, not during RL training.

---

## 5. Stable Training Recipe: From Environment to Hyperparameters

### 5.1 Context structure and reasoning paradigm

We use the rLLM [[1](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] default multi-turn Tool-Agent (ReAct) pattern. A trajectory has the following structure:

1. `system`: rules + tool descriptions (including the search tool format)
2. `user`: question
3. Repeated alternation of:
    - `assistant`: reasoning + tool calls (potentially multiple `search` calls per step);
    - `tool`: top-k document snippets / summaries returned by the retrieval server.
4. Final `assistant`: the model’s final answer.

As a baseline, we use the simplest setup:

- Provide only a `search` tool: given a query, return the top-10 most similar document snippets in the corpus (truncated).
- Use an auxiliary LLM to summarize these snippets and return only the summary (see Section 6).
- Do not provide open-doc / Python / Web tools.

### 5.2 Sampling configuration (rollout)

We follow the sampling configuration used in several search-agent papers:

- **temperature = 1.0**
- **top_p = 1.0**

Other key settings:

- **Maximum search steps (training)**: 25
    - A higher step limit tends to amplify noise and makes training less stable;
    - Relax it to 48 / 64 later in training; for validation, it is useful to consistently set it to 64 to inspect the ceiling.
- **`max_tokens_per_traj` ≈ 32k** (including the initial prompt + tool outputs + model outputs)
- **Reject_sampling = True**

### 5.3 GRPO training hyperparameters

Core training configuration:

- **Algorithm**: GRPO (Group Relative Policy Optimization) [[5](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]
- **group size**: 8
- **batch_size**: 128 (total `128×8=1024` trajectories)
- **ppo_mini_bs**: 32 (total `32×8=256` trajectories)
- **learning_rate**: `1e-6`
- **grad_clip**: `1.0`
- **reward**: `F1(pred_answer, gold_answer)`

### 5.4 Handling abnormal trajectories

During RL training for multi-turn search agents, it is possible to roll out abnormal trajectories, such as:

1. **A single `assistant` step emits a large burst of parallel tool calls** (e.g., 10+ searches in one message)
2. **Tool parse errors** (e.g., unclosed `<think>` tags or malformed JSON parameters in the model output)
3. **Repeated queries** (the current query in the model output exactly matches a previous query in the same trajectory)
4. **Search errors** (tool-call failures, such as retrieval timeouts)
5. **Exceeding `max_steps` / exceeding the token budget**

These abnormal trajectories often have more extreme PPO importance ratios; without specific handling, they amplify gradient noise and destabilize training. We categorize abnormalities as either model-caused or environment-caused, and apply a unified, reproducible handling policy.

Handling policy:

### 5.4.1 Break + 0 reward

For the following cases, once triggered we immediately stop the trajectory and assign **0 reward**:

- The number of tool calls in a single step exceeds a threshold (e.g., > 5 or > 10);
- A tool parse error occurs in the model output;
- The current search query in the model output exactly matches any previous query in the trajectory.

In practice, this significantly reduces the model’s abnormal behavior frequency and is more stable than “penalize only the final message” variants (partial masking can create abnormal sample weighting under some loss aggregations).

### 5.4.2 Search errors: discard directly

For search errors caused by the environment (timeouts, etc.):

- Discard the trajectory completely;
- Only record its frequency in monitoring metrics to debug environment stability.

### 5.4.3 Exceeding the search step limit: stop + 0 reward

For samples that hit `max_steps`:

- We stop rollout at that point;
- We assign 0 reward.

Trajectories with very high step counts tend to have a higher abnormal-rate; stopping them with 0 reward both reduces gradient pollution and suppresses redundant search.

### 5.4.4 Exceeding the token budget: compute advantage, exclude from updates

For samples that are truncated due to hitting the token limit:

- Still use them to compute group-wise advantages (to reduce statistical bias);
- But **exclude them from the loss and backpropagation**.

### 5.5 Framework fixes: TITO & TIS

rLLM’s RL training stack uses vLLM rollouts + Megatron training to accelerate training. In multi-turn Tool-Agent settings, two mismatch issues require special attention.

### 5.5.1 TITO: aligning rollout and training token sequences

rLLM’s agent loop converts between tokens and strings. In practice, decode+encode can change tokens, especially for sequences with abbreviations (and interestingly, the token corresponding to the word “alternatively” can split into three tokens after a decode+encode round-trip). This makes rollout-side and training-side token sequences inconsistent; if applying TIS [[16](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], the ratio can be computed on misaligned tokens, which hurts training stability.

The fix is simple: **strict token-in-token-out (TITO)**. Pass the response tokens generated by vLLM directly to the Megatron training engine as token sequences; only decode when displaying or logging.

### 5.5.2 TIS: aligning training and inference distributions

**The log probabilities from vLLM rollouts and from Megatron training are not perfectly aligned** (more pronounced for MoE). The community has proposed many solutions [[16](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[17](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[18](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[19](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)]. We adopt the TIS method proposed in [[16](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] to address this issue.

---

## 6. Refine Agent: Summarizing Retrieval Results with an Auxiliary LLM

### 6.1 Problem: raw documents are long and noisy

If we directly feed the raw documents retrieved from the server into the search agent, several issues arise:

1. **Too many tokens.** If snippets are short, they may not contain enough information for the model to infer the answer; if snippets are long, the context window gets exhausted after a few search steps and truncation becomes frequent.
2. **Messy distribution.** Documents contain HTML, tables, rare symbols, etc., which introduces noise; the model is more likely to generate irrelevant content.

### 6.2 Approach: integrate the refine agent into the tool

Following the design in [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21),[7](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], we summarize retrieved documents with an auxiliary model. In our experiments, this improves context utilization and significantly boosts training stability, so we recommend it as part of the baseline pipeline.

1. Use the retrieval server to fetch raw documents;
2. Feed “raw documents + current query” into an auxiliary model (e.g., Qwen3-8B);
3. The refine agent outputs a query-relevant summary;
4. The tool returns only the summary to the main search agent.

The refine agent:

- Does not participate in RL training (fixed pretrained weights);
- Runs as a separate vLLM server and is called over HTTP;
- Appears as a black box to the main agent, which only sees “shorter and cleaner” tool outputs.

### 6.3 Empirical observations

Compared with runs without a refine agent:

1. **Trajectory length becomes more controllable**: with the same 32k context budget, longer retrieval chains are supported and truncation drops;
2. **Training stability improves**;
3. **Generalizes well**: on HotpotQA / Bamboogle, performance with and without the refine agent is similar at evaluation time.

---

## 7. Metrics and Monitoring: Detecting Policy Regression Early

When training Search Agents, monitoring only loss / reward is often insufficient for early detection of policy regression or training collapse. It is more effective to also monitor behavior metrics, abnormal-rate metrics, and optimization-stability metrics.

### 7.1 Search-related metrics

Track the following for **all trajectories / correct trajectories**:

- Average number of search steps (number of `assistant` messages);
- Average number of search calls (total tool calls per trajectory);
- The median and maximum of the metrics above.

### 7.2 Abnormal-sample metrics

Track the proportion of each abnormal type in a batch:

- **Repeated queries**: without handling can reach ~30%; with abnormal handling can drop to ~1%;
- **Tool parse errors**: without handling can reach ~10%; with abnormal handling can drop to <1%;
- **Hitting `max_steps`**: typically under 10%;
- **Truncated by token budget**: typically under 5%;
- **Burst tool calling in a single step**: typically under 10%;
- **Non-English characters (especially Chinese)**: a sensitive early warning signal on English-only datasets; in some failed runs, `cjk_rate` increases before `grad_norm` or reward deteriorates.

### 7.3 Training-related metrics

More direct indicators of optimization stability include:

- **PPO importance ratio quantiles**: p5 / p95, and the fraction of tokens beyond clipping;
- **TIS ratio quantiles**: p5 / p95 / p100, and the fraction of tokens beyond a threshold; the minimum can be as small as `1e-15`, while a maximum significantly above 10 typically warrants prioritizing token-mismatch debugging;
- **grad_norm**: sustained increases or spikes can indicate policy regression or training collapse;
- **Reject-sampling ratio / number of effective groups**: helps track changes in effective batch size.

---

## 8. Common Pitfalls

### 8.1 Fallback logic for abnormal model behavior

To reduce code complexity, some frameworks use coarse default handling for abnormal behaviors. For example:

- If a tool call has an invalid format: It may return an empty message or stop the trajectory.
- If tool parameters are invalid: It may directly raise an exception and restart the entire trajectory.

I recommend explicitly taking control of fallback logic to ensure “controlled failure”:

- For all errors caused by the model, handle them via break + 0 reward whenever possible;
- For errors caused by the environment (timeouts, etc.), discard the trajectories and do not include them in updates.

### 8.2 Concurrency control

The default tool-calling logic lacks strict concurrency control, which causes at least two problems:

1. If some trajectories issue a “burst” of tool calls in one step, it can easily **exceed the system’s thread limits**;
2. Without an explicit concurrency cap, the retrieval server is under heavy load:
    - Search success rate decreases over time;
    - When a single abnormal trajectory spawns a large number of concurrent searches, it can flood the retrieval server and **hurt the search success rate of the entire batch**.

In practice, we recommend:

- **Adding explicit concurrency control at the tool-calling layer**:
    - Set the maximum concurrent search limit for a single trajectory in the same round;
    - Configure a maximum concurrency for each tool’s call function, trading waiting time for success rate;
    - Dynamically tune the concurrency limit and timeout based on observed search success rate and latency.

---

## 9. Experimental Results (8B & 30B-A3B)

### 9.1 Qwen3-8B trained for 300 iterations

| Dataset | EM | F1 | Avg. steps per correct trajectory |
| --- | --- | --- | --- |
| BrowseComp-Plus | 0.30 | 0.36 | 13.1 |

The following figures show the pass@1 curve of the 8B model on the test set, the training reward curve, and the tool call step counts during the training/testing phases with iterations:

![8B validation EM curve](attachment:eee0bdd8-ac22-4cdd-b088-bfdfae1a6a9a:en8B测试集EM曲线.png)

![8B training reward curve](attachment:8a7798f1-b646-4a38-af2f-f64a3aa84649:en8B训练reward曲线.png)

![8B tool call step counts - training](attachment:c190f1a7-ee3b-44c0-a197-80179afa318f:en8B工具调用曲线-训练.png)

![8B tool call step counts - evaluation](attachment:d9dc260d-a6f9-496a-84db-3fdedc5d3411:en8B工具调用曲线-测试.png)

### 9.2 Qwen3-30B-A3B-Thinking-2507 trained for 100 iterations

| Dataset | EM | F1 | Avg. steps per correct trajectory |
| --- | --- | --- | --- |
| BrowseComp-Plus | 0.29 | 0.35 | 18.9 |
| HotpotQA | 0.495 | 0.630 | 4.7 |
| Bamboogle | 0.632 | 0.724 | 3.5 |
| 2WikiMQA | 0.496 | 0.587 | 7.7 |
| Musique | 0.239 | 0.371 | 6.6 |

The following figures show the pass@1 curve of the 30B-A3B model on the test set, the training reward curve, and the tool call step counts during the training/testing phases with iterations:

![A3B validation EM curve](attachment:f15c57b2-35f2-4d21-b8a5-b78b7c73b331:enA3B测试EM曲线.png)

![A3B training reward curve](attachment:3ebc8243-e917-443d-9e5f-0a9a105875b1:enA3B训练reward曲线.png)

![A3B tool call step counts - training](attachment:9925023a-a8b4-49cd-8759-f8a6ac28de99:enA3B工具调用曲线-训练.png)

![A3B tool call step counts - evaluation](attachment:9704e1a8-3efa-4908-ba76-261db0fc8c76:enA3B工具调用曲线-测试.png)

### 9.3 Impact of Filtering Abnormal Trajectories

Sections [9.1](about:blank#section-91) and [9.2](about:blank#section-92) report results without the training strategy described in Section [5.4.1](about:blank#section-541). After enabling it, convergence becomes slightly slower but the final ceiling remains the same, trajectory quality improves noticeably, and the rate of repeated searches on the test set drops from about 20% to below 2%. The figure below plots the training curves of the 8B model after adopting the abnormal trajectory penalty.

![8B early break validation EM curve](attachment:c656a4d5-6e56-4cd5-a800-98ca129317df:en8Bearlybreak策略测试集EM曲线.png)

![8B early break tool call step counts](attachment:ecd4b18e-f811-4a2b-93f5-f65086ca9ed4:en8Bearlybreak策略测试集工具调用曲线.png)

### 9.4 Behavioral Observations

We also inspected how the model’s behavior evolves during training and found several patterns that help explain why both the number of search turns and the final accuracy improve through RL training:

- **Answers increasingly originate from retrieved evidence instead of prior knowledge.** At the early training stages, the models mostly guess based on their pretraining knowledge and only perform ceremonial searches. As training continues, answers rely more on retrieved results. To quantify this, we track where the final answer first appears in a trajectory. If it first shows up in the model’s output, the model was guessing; if it first appears in the tool response, the model drew from search results. The curve below shows the probability shift toward the tool outputs as RL progresses.
    
    ![First appearance of answers during evaluation](attachment:d5edfa65-4b07-48ff-8a9a-09d688abab49:enanswer_first_presented.png)
    
- **Tokens related to retrieval, multi-turn dialogue, expressing uncertainty, and logical reasoning become more frequent.** We compute a moving average of token frequencies throughout training and visualize the most increased ones. The figure shows that retrieval- and reasoning-related tokens trend upward, indicating stronger multi-turn retrieval plus reasoning capabilities.
    
    ![Tokens with increasing frequency during evaluation](attachment:926a3849-9ebd-4abf-a1c3-fdcdff8bd891:entoken_vis.png)
    
- **Search strategies shift from long natural language queries to concise keyword queries.** Early models often submit long, noisy sentences that contain too many keywords, yielding low-quality hits. Later checkpoints learn to decompose the task and issue shorter keyword searches. The following plot tracks the average character length of each query and shows a clear downward trend.
    
    ![Average query length trend during evaluation](attachment:945452ca-ad23-407a-a1aa-20ce1003193b:enquery_length_trend.png)
    

---

## 10. Potential Directions for Further Improvement

This post provides an engineering-oriented, reproducible starting point. Further improvements can consider:

1. **Introduce PRMs to suppress redundant search**, as in [[20](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
2. **Transition from the ReAct paradigm to an MDP formulation**, as in [[9](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] and [[21](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
3. **Improve the loss function**, for example with GSPO [[22](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], MiniRL [[23](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], or SAPO [[24](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
4. **Factorize agent responsibilities**, as explored in [[21](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
5. **TTS (Trajectory Token Selection / Truncation)**, related work includes [[10](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], [[11](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)], [[25](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
6. **Context management**, see [[26](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
7. **Partial rollout**, as in [[4](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)].
8. **Aggressive oversampling**: in practice, the stability of the number of effective groups strongly influences training stability. To improve oversampling efficiency, you can follow [[11](https://www.notion.so/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c?pvs=21)] and dynamically adjust the dataset.

## 11. References

1. *rLLM: A Framework for Post-Training Language Agents*
2. *Qwen3 Technical Report*
3. *BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent*.
4. *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL*.
5. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*
6. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*
7. *Open Data Synthesis For Deep Research*
8. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*
9. *IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction*
10. *PokeeResearch: Effective Deep Research via Reinforcement Learning from AI Feedback and Robust Reasoning Scaffold*
11. *Tongyi DeepResearch Technical Report*
12. *WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents*
13. *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*
14. *MuSiQue: Multihop Questions via Single-hop Question Composition*
15. *BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents*
16. *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*
17. *When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch*
18. *Small Leak Can Sink a Great Ship—Boost RL Training on MoE with IcePop!*
19. *Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?*
20. *CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic*.
21. *In-the-Flow Agentic System Optimization for Effective Planning and Tool Use*
22. *Group Sequence Policy Optimization*
23. *Stabilizing Reinforcement Learning with LLMs: Formulation and Practices*
24. *Soft Adaptive Policy Optimization*
25. *AIA Forecaster: Experimenting with Agentic AI for News Event Forecasting*
26. Lost In The Maze: Overcoming Context Limitations In Long-Horizon Agentic Search
