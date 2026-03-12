# Configs

This tree is the canonical runnable surface for the repository.

Rule of thumb:

- `configs/` contains entrypoints you run
- `examples/` contains task logic, walkthroughs, and reference code

Config files should stay thin. They should wire together existing task code and
runtime settings, not re-implement scoring, parsing, or resource management.

Each config should expose a `config_status` object from
`rollouts.config_status` so we can distinguish:

- `draft`: written down, not validated
- `import_tested`: imports cleanly on the current codebase
- `known_good`: actually run and validated as of a specific commit
