# RLM Expressiveness Discussion

Source: https://x.com/lateinteraction/status/1878896620499075500
Date: 2025-01-13

## Omar Khattab (@lateinteraction)

On RLMs: The way people tend to implement recursive sub-calls or "sub-agents" don't work.

In particular, you cannot express sub-agents as tool calls.

To see why those aren't expressive enough, suppose you are given a 10M-token prompt that starts with "keep each math question below iff the answer is an even number".

A typical sub-agent strategy would fail on a few angles:

1) It can't *write* the O(N) sub-calls as tool calls, because the model can't verbalize that many explicit sub-prompts. In other words, the recursion has to be symbolic through code, not tool calls.

2) It also can't fit the long prompt in the first place—so it may resort to compaction. In other words, it's crucial for an RLM that its own prompts/requests (not just the typical external "environment") are accessible through pointers as an object, so it can recurse through them symbolically.

Just wanted to note these because I see people get excited then implement ideas that are much less expressive than RLMs are supposed to be.

---

## Tenobrus (@tenobrus)

any coding agent's scaffold can do this easily w tool calls + a filesystem by saving "the prompt" as a file in the filesystem, using "read and follow the instructions in x file path" as the actual prompt, and then using its own code + file + bash tools

---

## Omar Khattab (@lateinteraction)

Yes! If you build an RLM by (1) externalizing your prompt as an object, (2) allowing recursive calls in a coding sandbox, then you indeed end up with an RLM.

No one is saying it's *hard*. It's very easy in fact. But it's not how the models work, which is why CC does compaction.

---

## Tenobrus (@tenobrus)

fair nuf
