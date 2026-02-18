# pi-mono vs rollouts TUI Comparison

## Architecture Overview

| Aspect | pi-mono (TypeScript) | rollouts (Python) |
|--------|---------------------|-------------------|
| **Base Library** | Custom terminal abstraction | pytui (external dependency) |
| **Input Handling** | Centralized `StdinBuffer` with event emission | Direct terminal reads with manual parsing |
| **Paste Handling** | `stdin-buffer.ts` + bracketed paste re-wrapping | Direct marker detection in components |
| **Key Bindings** | Full `keybindings.ts` + `keys.ts` system | Hardcoded in input handling loops |
| **Kill Ring** | `kill-ring.ts` (Emacs-style cut/paste) | **MISSING** |
| **Undo Stack** | `undo-stack.ts` with structuredClone | **MISSING** |

---

## Detailed Issues Found

### 1. Paste Handling Issues

#### pi-mono approach (more robust):
```typescript
// stdin-buffer.ts handles paste at terminal level
export class StdinBuffer extends EventEmitter<StdinBufferEventMap> {
    private pasteMode: boolean = false;
    private pasteBuffer: string = "";
    
    // Emits separate 'paste' event
    this.emit("paste", pastedContent);
}

// Terminal re-wraps paste for backward compatibility
this.stdinBuffer.on("paste", (content) => {
    if (this.inputHandler) {
        this.inputHandler(`\x1b[200~${content}\x1b[201~`);
    }
});
```

#### rollouts approach (potential issues):
```python
# input.py - direct marker handling, no chunk buffering
if "\x1b[200~" in data:
    self._is_in_paste = True
    self._paste_buffer = ""
    data = data.replace("\x1b[200~", "")

if self._is_in_paste:
    self._paste_buffer += data
    end_index = self._paste_buffer.find("\x1b[201~")
    # ... handle paste
```

**Problems with rollouts approach:**
- No handling of paste markers split across multiple `read_input()` calls
- If `\x1b[200~` arrives in one chunk and content in another, paste detection fails
- No centralized paste buffering like pi-mono's `StdinBuffer`

---

### 2. Interrupt Handling Issues

#### pi-mono approach:
- Ctrl+C passed to focused component via `handleInput(data)`
- Component decides how to handle via keybindings: `selectCancel: ["escape", "ctrl+c"]`
- Clean separation of concerns

#### rollouts approach (problematic):
```python
# interactive_agent.py - _input_reading_loop
if len(input_data) > 0 and ord(input_data[0]) == 3:  # Ctrl+C
    if self.cancel_scope:
        self.cancel_scope.cancel()
    return

# tui.py - _handle_input
if len(data) > 0 and ord(data[0]) == 3:  # Ctrl+C
    return  # Just ignore it!

# Also has SIGINT handler that may conflict:
signal.signal(signal.SIGINT, self._handle_sigint)
```

**Problems:**
1. Multiple competing Ctrl+C handlers (input loop, TUI, signal handler)
2. TUI layer silently drops Ctrl+C instead of routing to components
3. No keybindings system - hardcoded everywhere
4. Escape key vs Ctrl+C distinction is fragile (relies on `escape_pressed` flag)

---

### 3. Missing Features

| Feature | pi-mono | rollouts | Impact |
|---------|---------|----------|--------|
| **Kill Ring** | Full implementation with accumulate/prepend | Missing | No Emacs-style cut/yank |
| **Undo Stack** | Generic with structuredClone | Missing | No undo in editor |
| **Keybindings System** | Full configurable system | Hardcoded | Can't customize shortcuts |
| **Kitty Protocol** | Supported in keys.ts | Not implemented | No key release detection |
| **Stdin Buffering** | Centralized sequence parsing | Relies on pytui | Partial sequences may break |

---

### 4. Editor Component Differences

#### pi-mono editor.ts features (rollouts missing):
- Word-wrap with proper grapheme handling
- Undo history with fish-style coalescing
- Kill ring for cut/yank operations
- Kitty keyboard protocol support
- Autocomplete integration
- Proper handling of CSI-u sequences

#### rollouts input.py limitations:
- Simple line-based layout (no word wrap)
- No undo/redo
- No kill ring
- Hardcoded key handling
- Basic paste with marker detection

---

## Specific Code Issues in rollouts

### File: `frontends/tui/components/input.py`

1. **Paste handling doesn't chunk properly**:
   ```python
   if "\x1b[200~" in data:  # Assumes marker arrives in single read
   ```
   If terminal sends `\x1b[200~` and paste content in separate reads, this breaks.

2. **No handling of paste markers within content**:
   - Doesn't handle case where paste content itself contains `\x1b[201~`
   - pi-mono's `StdinBuffer` handles this with proper state machine

### File: `frontends/tui/interactive_agent.py`

1. **Conflicting interrupt handlers**:
   ```python
   # Line 1086 - input loop handles Ctrl+C
   if len(input_data) > 0 and ord(input_data[0]) == 3:
   
   # Line 1236 - SIGINT handler also registered
   signal.signal(signal.SIGINT, self._handle_sigint)
   ```

2. **Escape detection is fragile**:
   ```python
   # Line 1106 - checks if focused component is input_component
   if input_data == "\x1b":
       if self.tui._focused_component is not self.input_component:
           # Route to focused component
   ```
   This bypasses TUI's normal input routing for Escape key only.

### File: `frontends/tui/tui.py`

1. **Ctrl+C silently dropped**:
   ```python
   # Line 745 - should route to component or allow override
   if len(data) > 0 and ord(data[0]) == 3:
       return  # Silent drop!
   ```

---

## Recommendations

### High Priority

1. **Fix paste handling** - Implement proper chunked paste detection similar to pi-mono's `StdinBuffer`
2. **Unify interrupt handling** - Remove competing Ctrl+C handlers, use single keybindings system
3. **Add keybindings abstraction** - Port `keybindings.ts` and `keys.ts` architecture

### Medium Priority

4. **Add undo support** - Port `undo-stack.ts` generic implementation
5. **Add kill ring** - Port `kill-ring.ts` for Emacs-style operations
6. **Improve editor capabilities** - Word wrap, better cursor handling

### Low Priority

7. **Kitty protocol support** - For better key event detection
8. **Input buffering layer** - Centralized escape sequence parsing

---

## Code Port Checklist

Files from pi-mono that could be ported:

- [ ] `stdin-buffer.ts` - Centralized input buffering with paste support
- [ ] `keybindings.ts` - Configurable key binding system
- [ ] `keys.ts` - Key sequence parsing and matching
- [ ] `kill-ring.ts` - Emacs-style kill/yank ring
- [ ] `undo-stack.ts` - Generic undo with structuredClone
- [ ] `utils.ts` - Grapheme segmentation, width calculation
