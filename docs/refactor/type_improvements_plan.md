# Type Improvements - Frozen Dataclasses

**Date**: 2025-10-12
**Status**: 🔄 Ready to Implement
**Estimated Time**: 1-2 hours

---

## Objective

Replace generic `Dict[str, str/Any]` with frozen dataclasses for domain types to improve:
1. **Type Safety**: Runtime validation + type checker support
2. **Immutability**: Prevent accidental mutations (functional programming)
3. **Tiger Style**: Add assertions in `__post_init__`
4. **Developer Experience**: IDE autocomplete, clear intent

---

## Dataclasses to Create

### 1. EnvironmentVariables

**Current Usage**: `env_vars: Optional[Dict[str, str]]` (10 occurrences)

**Files Affected**:
- `bifrost/bifrost/types.py` (new dataclass)
- `bifrost/bifrost/client.py` (exec, deploy, run_detached)
- `bifrost/bifrost/deploy.py` (deploy methods, make_env_payload)
- `bifrost/bifrost/job_manager.py` (start_tmux_session)

**Implementation**:
```python
@dataclass(frozen=True)
class EnvironmentVariables:
    """Environment variables for remote command execution.

    Validates variable names follow shell naming rules.
    Immutable to prevent accidental modification during execution.
    """
    variables: Dict[str, str]

    def __post_init__(self):
        # Tiger Style: assert all inputs
        assert isinstance(self.variables, dict), "variables must be dict"

        for key, value in self.variables.items():
            # Shell variable name rules: [A-Za-z_][A-Za-z0-9_]*
            assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key), \
                f"Invalid environment variable name: {key}"
            assert isinstance(value, str), \
                f"Environment variable {key} must be string, got {type(value)}"

        # Assert output invariant
        assert len(self.variables) >= 0, "variables dict created"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict for backward compatibility."""
        return self.variables.copy()

    @classmethod
    def from_dict(cls, variables: Optional[Dict[str, str]]) -> Optional['EnvironmentVariables']:
        """Create from optional dict (for gradual migration)."""
        if variables is None:
            return None
        return cls(variables=variables)
```

**Migration Strategy**:
- Add `from_dict()` classmethod for backward compatibility
- Update function signatures to accept `Optional[EnvironmentVariables]`
- Keep internal `.variables` dict for existing code
- Type checker will catch misuse

---

### 2. ProviderCredentials

**Current Usage**: `credentials: Dict[str, str]` (1 occurrence)

**Files Affected**:
- `broker/broker/types.py` (new dataclass)
- `broker/broker/client.py` (GPUClient.__init__)
- `broker/broker/api.py` (all api functions)

**Implementation**:
```python
@dataclass(frozen=True)
class ProviderCredentials:
    """API credentials for cloud GPU providers.

    Supports multiple providers (RunPod, Vast, Lambda, etc).
    Immutable to prevent accidental credential leaks.
    """
    runpod: str = ""
    vast: str = ""
    # Add more providers as needed

    def __post_init__(self):
        # Tiger Style: assert at least one credential provided
        assert self.runpod or self.vast, \
            "At least one provider credential required"

        # Validate credential format (basic length check)
        if self.runpod:
            assert len(self.runpod) > 10, \
                "RunPod API key appears invalid (too short)"

        if self.vast:
            assert len(self.vast) > 10, \
                "Vast API key appears invalid (too short)"

        # Assert output invariant
        assert self.runpod or self.vast, "credentials validated"

    def get(self, provider: str) -> Optional[str]:
        """Get credential for specific provider."""
        if provider == "runpod":
            return self.runpod
        elif provider == "vast":
            return self.vast
        return None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict for backward compatibility."""
        result = {}
        if self.runpod:
            result["runpod"] = self.runpod
        if self.vast:
            result["vast"] = self.vast
        return result

    @classmethod
    def from_dict(cls, credentials: Dict[str, str]) -> 'ProviderCredentials':
        """Create from dict (for backward compatibility)."""
        return cls(
            runpod=credentials.get("runpod", ""),
            vast=credentials.get("vast", "")
        )
```

**Migration Strategy**:
- Add both `from_dict()` and `to_dict()` for compatibility
- Update GPUClient to use dataclass
- Update api.py functions to accept ProviderCredentials
- Add `.get(provider)` method for internal lookups

---

### 3. SessionInfo

**Current Usage**: `get_session_info() -> Dict[str, str]` (1 occurrence)

**Files Affected**:
- `bifrost/bifrost/types.py` (new dataclass)
- `bifrost/bifrost/client.py` (get_session_info)

**Implementation**:
```python
@dataclass(frozen=True)
class SessionInfo:
    """Information about tmux sessions for a detached job.

    Provides session names and SSH commands to attach to them.
    Immutable since session info shouldn't change after creation.
    """
    job_id: str
    main_session: str
    attach_main: str
    bootstrap_session: Optional[str] = None
    attach_bootstrap: Optional[str] = None

    def __post_init__(self):
        # Tiger Style: assert inputs and invariants
        assert len(self.job_id) > 0, "job_id cannot be empty"
        assert self.main_session.startswith("bifrost-"), \
            f"Invalid main session name: {self.main_session}"
        assert "ssh" in self.attach_main, \
            "attach_main must be SSH command"

        # Validate bootstrap session format if present
        if self.bootstrap_session:
            assert self.bootstrap_session.endswith("-bootstrap"), \
                f"Invalid bootstrap session name: {self.bootstrap_session}"
            assert self.attach_bootstrap is not None, \
                "attach_bootstrap required when bootstrap_session present"

        # Assert output invariant
        assert self.main_session, "session info validated"
```

**Migration Strategy**:
- Direct replacement - return dataclass instead of dict
- No backward compatibility needed (new API in Phase 3)
- Type checker provides full IDE support

---

### 4. JobMetadata

**Current Usage**: `create_job_metadata() -> Dict[str, Any]` (1 occurrence)

**Files Affected**:
- `bifrost/bifrost/types.py` (new dataclass)
- `bifrost/bifrost/job_manager.py` (create_job_metadata)

**Implementation**:
```python
@dataclass(frozen=True)
class JobMetadata:
    """Metadata for a detached job execution.

    Stored in ~/.bifrost/jobs/{job_id}/metadata.json on remote.
    Immutable since metadata is write-once, read-many.
    """
    job_id: str
    command: str
    ssh_info: str
    status: str
    start_time: str  # ISO 8601 format
    tmux_session: str
    worktree_path: str
    git_commit: str
    repo_name: str
    end_time: Optional[str] = None  # ISO 8601 format
    exit_code: Optional[int] = None

    def __post_init__(self):
        # Tiger Style: assert all inputs and invariants
        assert len(self.job_id) > 0, "job_id cannot be empty"
        assert len(self.command) > 0, "command cannot be empty"

        # Validate status
        valid_statuses = {"starting", "running", "completed", "failed", "killed"}
        assert self.status in valid_statuses, \
            f"Invalid status: {self.status}, must be one of {valid_statuses}"

        # Validate git commit hash (full SHA-1 or SHA-256)
        assert len(self.git_commit) in (40, 64), \
            f"Invalid git commit hash length: {len(self.git_commit)}"

        # Validate SSH info format
        assert "@" in self.ssh_info and ":" in self.ssh_info, \
            f"Invalid ssh_info format: {self.ssh_info}"

        # Validate exit code range if present
        if self.exit_code is not None:
            assert 0 <= self.exit_code <= 255, \
                f"Invalid exit code: {self.exit_code}"

        # Assert output invariant
        assert self.job_id, "job metadata validated"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "job_id": self.job_id,
            "command": self.command,
            "ssh_info": self.ssh_info,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "tmux_session": self.tmux_session,
            "worktree_path": self.worktree_path,
            "git_commit": self.git_commit,
            "repo_name": self.repo_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadata':
        """Create from dict (for JSON deserialization)."""
        return cls(
            job_id=data["job_id"],
            command=data["command"],
            ssh_info=data["ssh_info"],
            status=data["status"],
            start_time=data["start_time"],
            tmux_session=data["tmux_session"],
            worktree_path=data["worktree_path"],
            git_commit=data["git_commit"],
            repo_name=data["repo_name"],
            end_time=data.get("end_time"),
            exit_code=data.get("exit_code")
        )
```

**Migration Strategy**:
- Add `to_dict()` for JSON serialization
- Add `from_dict()` for JSON deserialization
- Update create_job_metadata() to return dataclass
- Update metadata.json read/write to use dataclass

---

## Types to Keep as Dict

### LoggerLevels
**Keep**: `Dict[str, str]` - simple mapping, no structure

### Logging Config
**Keep**: `config: Dict[str, Any]` - Python stdlib format

### Raw Data
**Keep**: `raw_data: Optional[Dict[str, Any]]` - intentionally unstructured

### External API Responses
**Keep**: `Dict[str, Any]` - external API format, not our control

---

## Implementation Order

1. **Create dataclasses** in `types.py` files (no breaking changes yet)
2. **Add helper methods** (`from_dict`, `to_dict`) for compatibility
3. **Update function signatures** one by one
4. **Run type checker** after each change
5. **Test examples** to ensure nothing breaks

### Phase 1: Add Dataclasses (Non-Breaking)
```bash
# Add to bifrost/bifrost/types.py
- EnvironmentVariables
- SessionInfo
- JobMetadata

# Add to broker/broker/types.py
- ProviderCredentials
```

### Phase 2: Update Signatures (Breaking Changes)
```bash
# Update one module at a time:
1. broker/broker/client.py - ProviderCredentials
2. broker/broker/api.py - ProviderCredentials
3. bifrost/bifrost/client.py - EnvironmentVariables, SessionInfo
4. bifrost/bifrost/deploy.py - EnvironmentVariables
5. bifrost/bifrost/job_manager.py - EnvironmentVariables, JobMetadata
```

### Phase 3: Update Examples
```bash
# Fix examples to use new types:
- examples/phase1_example.py
- examples/phase3_example.py
```

### Phase 4: Run Type Checker
```bash
uv run ty check  # Should pass with 0 errors
```

---

## Testing Strategy

**Type Checker**: Must pass `ty check` with 0 errors/warnings

**Runtime Validation**: Assertions will catch invalid data at runtime

**Examples**: Run both examples to verify no regressions:
```bash
# Should work with new types
python examples/phase1_example.py --dry-run
python examples/phase3_example.py --dry-run
```

---

## Benefits After Implementation

1. ✅ **Type Safety**: IDE autocomplete for all domain types
2. ✅ **Runtime Validation**: Tiger Style assertions catch bugs early
3. ✅ **Immutability**: `frozen=True` prevents accidental mutations
4. ✅ **Documentation**: Dataclass fields are self-documenting
5. ✅ **Refactoring**: Type checker catches breaking changes
6. ✅ **Code Quality**: Clear intent vs generic Dict[str, Any]

---

## Risk Mitigation

**Backward Compatibility**:
- `from_dict()` classmethod for gradual migration
- `to_dict()` method for existing dict-based code
- Optional fields with defaults to avoid breaking changes

**Type Checker Issues**:
- Add `# type: ignore` only if absolutely necessary
- Prefer fixing the issue properly vs suppressing

**Runtime Errors**:
- Assertions provide clear error messages
- Better to fail fast than corrupt state

---

## Success Criteria

- [  ] All 4 dataclasses created in types.py files
- [  ] All function signatures updated
- [  ] `uv run ty check` passes with 0 errors
- [  ] Examples run successfully
- [  ] No runtime errors in smoke tests
- [  ] Code is more readable and maintainable

**Estimated Time**: 1-2 hours
**Risk Level**: Low (backward compatibility via helper methods)
