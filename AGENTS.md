# Local Agent Notes

## Testing

- Prefer integration tests at stable cut points over granular mocked unit tests.
- Do manual testing first for new workflows, then crystallize that usage path into a trusted test.
- For bugs, prefer a minimized regression test that fails if the fix is reverted.
- Keep the end-to-end suite small and curated.

See:
- [grugbrain_testing.md](/Users/chiraagbalu/research/docs/code_style/grugbrain_testing.md)
- [code_philosophy_essay.md](/Users/chiraagbalu/research/docs/code_style/code_philosophy_essay.md)
- [debugging_draft.md](/Users/chiraagbalu/research/docs/code_style/debugging_draft.md)
