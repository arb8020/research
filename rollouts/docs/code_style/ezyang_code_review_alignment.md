# Code Review as Human Alignment in the Era of LLMs

**Source:** https://blog.ezyang.com/2025/12/code-review-as-human-alignment-in-the-era-of-llms/
**Author:** Edward Z. Yang
**Date:** December 2025

## Core Thesis

Code review in the LLM era should prioritize **human alignment** — ensuring the developer and reviewer share understanding of system design principles and architectural decisions — rather than focusing on mechanical code fixes.

## The Alignment Problem

When reviewing LLM-generated code, the real issue isn't just fixing bad patterns (like unnecessary try-catch blocks). The deeper problem:

> "The human who was operating the LLM didn't agree with me that this defensive code was bad."

The review comment isn't really for the LLM—it's to align the human developer's mental model.

## Shifted Expectations

Developers using LLMs should pre-align their tool's output before requesting review:

> "We should expect _more_ out of the author of code, because the cost of generating these PRs has gone way down."

## Reweighted Engineering Skills

LLM coding changes which competencies matter most. Now prioritized:
- Ability to read code
- Reasoning about the big picture
- Clear communication

De-prioritized:
- Mechanical coding ability

## The Bigger Picture

> "LLMs have no memory" — they must rediscover system context repeatedly.

Humans must "collectively maintain a shared vision of what the system should do." This is the true purpose of code review in the LLM era.

## Implication

Code review becomes less about catching bugs and more about maintaining architectural coherence and ensuring all human developers share the same mental model of the system.
