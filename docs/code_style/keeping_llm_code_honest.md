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

**lazy section of assorted tips**
- code smells to watch for: try/catches everywhere, deeply nested code, really long functions, functions with lots of args, classes used unnecessarily, unclear names. These are usually signs the code is lying about complexity. For functions with many arguments, there's often a shared context (user_id, tenant_id, request_context, logger, db_conn) that can be grouped into a struct that remains stable across your data transformation pipeline - this makes the actual transformation logic more visible.
- **task categorization:** bug fixes, refactors, and features each need different verification approaches. for features, manually test and feel out usage before crystallizing with tests. for bugs/refactors, tests already exist so focus on making code more honest about the problem
- **synchronous vs asynchronous usage:** use synchronous collaboration for surfacing code, discussing directions, modeling problems. use asynchronous for executing once you've defined the desired behavior and data flow
- git worktrees and branches are enough to get multiple LLM agents running at once while not stepping on each others toes. u can multiple chat sessions literally in parallel exploring different solution spaces, or solving different problems. any merge conflicts are also design feedback on how you might better structure the code to accomadate both features/bug fixes at once.
- create verification loops + logging setups where the LLM can efficiently run a command, jq or grep over the output logs, and figure out the undefined/unexpected behavior. This usually looks like standalone scripts with heavy side effect logging out to files - you're creating a custom observability layer the LLM can actually parse and reason about, not just "test failed" messages
- separate the 'understand codebase context' task as a different session from the 'execute patch' session when working on a difficult problem. if you can't oneshot it, you're probably closer to the edge of you + the LLMs coding ability so you might want to be more diligent abt ctx management
- code style docs help me crystallize my taste opinions in a way that helps the llm make a better decision on its own if i get lazy, and also to reduce the delta between the working code it generated and the code i would like it to have generated. i often ask LLMs to read these while working on a problem
- 'lmk if you have any questions, stop if you get confused or things feel complicated' is cheap way to catch obvious things and get the LLM to sit in a frame where it defers to you instead of wasting tokens on the wrong stuff
	- if you've run an LLM for a long time and it gave you something different from your spec it probably was because your spec wasn't as good as you thought, and the LLM had to deviate to solve the problem
- create short handoff documents when you finish a task/running out of context with the high problem spec, file/line ranges to look at, useful testing commands, and potentially things tried that didn't work. this is to manage context when you start to run out or to avoid polluting a follow up task with unnecessary old context
    - ex
        - CONTEXT: Users getting logged out after 5 min instead of 30
        - FILES: auth/session.py:45-67, config/timeouts.py
        - TEST: pytest tests/auth/test_timeout.py::test_session_duration
        - TRIED: Updating SESSION_TIMEOUT didn't work (it's overridden elsewhere)
    - ex:
        - CONTEXT: need to support digital ocean as a GPU provider
        - FILES: providers/base.py, providers/runpod.py
        - TESTS: none yet, check runpod provider for any integration tests
- inline comments/docstrings about gotchas are the best way to 'continual learn' in a codebase. injects the relevant unclear context when necessary.
    - note this is typically a code smell and you should probably write that section of code better. you shouldn't need to 'onboard' onto a codebase. names should be clear, program state should be easy to reason about, and users shouldn't have to simulate code paths to understand what's going on

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
