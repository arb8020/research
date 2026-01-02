# Your Job Is to Deliver Code You Have Proven to Work

**Source:** https://simonwillison.net/2025/Dec/18/code-proven-to-work/
**Author:** Simon Willison
**Date:** December 18, 2025

## Core Principle

> "Your job is to deliver code you have proven to work."

Junior developers using AI tools often submit untested code to reviewers, shifting the burden of validation onto others. This is a failure of professional duty.

## Two Required Steps for Proof

### 1. Manual Testing

- Personally verify the code works before submission
- Document the process as terminal commands with output, or screen recordings for visual changes
- Progress from happy path testing to edge cases
- Manual testing is a skill that distinguishes senior engineers

### 2. Automated Testing

- Bundle changes with tests that fail if the implementation is reverted
- Mirror manual testing: establish initial state, exercise change, assert results
- No excuse to skip this step given modern LLM tooling

## On Coding Agents

Teach AI coding tools to prove their work through:
- One-off testing capabilities
- Screenshot verification for visual changes
- Writing comprehensive test suites matching project patterns

## Key Insight

> "Almost anyone can prompt an LLM to generate a thousand-line patch... What's valuable is contributing code that is proven to work."

Human accountability remains essential—computers cannot be held responsible. Developers must provide evidence their code functions correctly before requesting review.
