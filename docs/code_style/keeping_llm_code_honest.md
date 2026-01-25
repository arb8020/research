---
layout: post
title: "keeping LLM code honest"
date: 2026-01-11
slug: keeping-llm-code-honest
---

i don't like to write boilerplate. as LLMs have gotten better, more and more code output feels like writing boilerplate. as a result, i've been daily driving coding with LLMs for about a year now. i'd estimate about 99% of the LoC i've produced that actually ran in prod since then has been LLM-generated.

for any given problem, there is a infinitely wide space of programs that will solve that problem. your job as an engineer is to choose the best solution in that space, based on your understanding of the problem. this is not necessarily something like least complex/least lines of code/other dumb proxy metrics. its just 'will this be easy for someone else to understand/modify later' in that it communicates the problem in its solution. so your job is to first ship working code with the LLM, then compress it into the right code that models the problem.

the right code almost explains the problem to you. bad code is dishonest about complexity or about problem shape. this can either be that its overly simple, and hides complexity where it shouldn't or that it adds undeserved complexity. for example, there may be unnecessary defensive checks in the hotpath of some code runtime, when instead you could have parsed external API into types that your code can trust. or perhaps you've written your runtime logic in a way that mixes concerns, putting filtering and work functions together when they could be separated.

LLMs write this bad code by default bc they're trained on a lot of bad code, and because they don't have the full problem context in the way that you do. an example is that they frequently write code that is hard to reason about because they use classes for both namespacing and state, when pure functions and types can make it easier to understand and constrain program state. it doesn't tend to write in a style that makes its own life easier when left unchecked.

because of this, there are domains where it honestly might be easier to go write the control flow yourself. legitimately difficult runtime logic (games, systems, compilers), often cause LLMs to fall over because they write code that works, not code that models the problem. genuinely complex problems can still be modeled honestly, but its better to assume your code can be simpler. remember that friction is feedback. if its hard for you or the LLM to write the code, someone fucked up somewhere. your code is lying about complexity somewhere.

---

references:

APIs and code compression
https://caseymuratori.com/blog_0015
https://caseymuratori.com/blog_0016
https://caseymuratori.com/blog_0025

system design
https://www.seangoedecke.com/good-system-design/

mechanical code tips to keep you out of trouble
https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md

observability
https://loggingsucks.com/

code review + your responsibility with LLM code
https://simonwillison.net/2025/Dec/18/code-proven-to-work/
https://blog.ezyang.com/2025/12/code-review-as-human-alignment-in-the-era-of-llms/
