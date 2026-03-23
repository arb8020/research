# Configs

This tree is the canonical runnable surface for the repository.

Rule of thumb:

- `configs/` contains entrypoints you run
- `examples/` contains task logic, walkthroughs, and reference code

Config files should stay thin. They should wire together existing task code and
runtime settings, not re-implement scoring, parsing, or resource management.

Current recommended starting points for RL live in:

- [trusted/rl/README.md](/Users/chiraagbalu/research/rollouts/configs/trusted/rl/README.md)

Stubbed canonical environment-family example shapes live in:

- [trusted/environment_examples/README.md](/Users/chiraagbalu/research/rollouts/configs/trusted/environment_examples/README.md)

Each config should expose a `config_status` object from
`rollouts.config_status` so we can distinguish:

- `draft`: written down, not validated
- `import_tested`: imports cleanly on the current codebase
- `known_good`: actually run and validated as of a specific commit
