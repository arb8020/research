# TUI/UX Gap Analysis: pi-mono vs rollouts

This document catalogs UI/UX implementation features in pi-mono's TUI that rollouts lacks. Complements `docs/PI_MONO_FEATURES.md` which covers agent/extensibility features.

## Feature Gaps

### 1. Terminal Image Rendering

**pi-mono**: Full image rendering via Kitty Graphics Protocol and iTerm2 Inline Images with automatic protocol detection (Kitty, Ghostty, WezTerm, iTerm2).

**rollouts**: No image rendering support.

**Impact**: Can't display screenshots, diagrams, generated charts, or vision model outputs inline.

---

### 2. Overlay/Modal System

**pi-mono**: Sophisticated overlay stack with:
- 9 anchor positions (center, top-left, top-center, etc.)
- Percentage-based sizing (`width: "50%"`)
- Min/max constraints (`minWidth: 20, maxHeight: "80%"`)
- Margin control per-side
- Visibility callbacks for responsive behavior
- Focus restoration when overlays close

**rollouts**: No overlay system—components render in strictly sequential flow.

**Impact**: Can't show floating dialogs, confirmation modals, or settings panels without disrupting message flow.

---

### 3. Settings/Toggle List Component

**pi-mono**: `SettingsList` component with:
- Value cycling (tap to toggle through options)
- Submenu support for nested settings
- Optional search/filter
- Per-item descriptions

**rollouts**: No settings list—configuration is CLI flags only.

---

### 4. Fuzzy Matching/Search

**pi-mono**: Dedicated fuzzy algorithm with:
- Scoring: consecutive matches, word boundaries, early matches rewarded
- Digit/letter swapping for flexible matching (`ab1` matches `1ab`)
- Multi-token support (space-separated, all must match)

**rollouts**: Prefix matching only in tab completion.

---

### 5. Path Autocomplete with Git Integration

**pi-mono**: File path completion with:
- `@"path"` syntax for file references
- Automatic quoting for paths with spaces
- Respects `.gitignore` via `fd`
- Tilde expansion

**rollouts**: Tab completion exists but no git-aware path completion.

---

### 6. Hyperlink Support

**pi-mono**: OSC 8 terminal hyperlinks—`[text](url)` renders as clickable links in supported terminals.

**rollouts**: Links render as underlined text + URL, not clickable.

---

### 7. Comprehensive Keybinding System

**pi-mono**:
- 20+ configurable editor actions
- Kill ring with yank-pop cycling (Emacs-style)
- Word navigation with punctuation awareness
- Key release event support (Kitty protocol)
- Full `EditorKeybindingsConfig` for customization

**rollouts**: Basic fixed keybindings (arrows, Ctrl+C, Ctrl+G, Tab).

---

### 8. Undo System

**pi-mono**: Full undo stack with clone-on-push semantics for text editing.

**rollouts**: No undo support in editor.

---

### 9. Syntax Highlighting in Code Blocks

**pi-mono**: Language-specific syntax highlighting with 9 token types:
- comment, keyword, function, variable, string, number, type, operator, punctuation

**rollouts**: Code blocks have borders but no syntax highlighting.

---

### 10. Extended Theme System

**pi-mono**: 51+ color definitions including:
- Diff colors (added/removed/context)
- 9 syntax highlighting colors
- 6 thinking intensity level colors
- Mode indicators (bash mode)
- Theme hot-reloading with cache invalidation

**rollouts**: ~15 theme colors, no diff/syntax/thinking colors.

---

### 11. Dynamic Border Component

**pi-mono**: `DynamicBorder` that adapts width to terminal, with theme-customizable colors.

**rollouts**: Static border rendering.

---

### 12. Countdown Timer Component

**pi-mono**: Visual countdown with per-second tick callbacks, auto-expiration.

**rollouts**: Shows delay in retry error text but no animated countdown.

---

### 13. Extension/Widget API

**pi-mono**: Full extension UI context:
- Dialog methods: `select()`, `confirm()`, `input()`, `editor()`
- Status methods: `setStatus()`, `setWorkingMessage()`, `notify()`
- Widget placement: `setWidget()` above/below editor
- Component replacement: `setEditorComponent()`, `setFooter()`, `setHeader()`
- Custom components: `custom()` with full overlay support

**rollouts**: No plugin/extension system for UI.

---

### 14. IME (Input Method Editor) Support

**pi-mono**: Full CJK input support with:
- `CURSOR_MARKER` APC sequences for candidate window positioning
- Hardware cursor control
- `Focusable` interface for focus tracking

**rollouts**: No explicit IME support.

---

### 15. Tree Navigation Component

**pi-mono**: `TreeSelector` for hierarchical navigation with parent/child relationships.

**rollouts**: No tree navigation component.

---

### 16. Interactive Session Selector

**pi-mono**: 35+ interactive session management components:
- Tree-based session navigation
- Search/filter
- Metadata display (tokens, branch, model)
- Rename, delete, fork operations

**rollouts**: Session ID in status line only—no interactive browser.

---

### 17. Configurable Tool Formatters

**pi-mono**: Custom tool formatters per tool type with render configs.

**rollouts**: Single unified tool display with 3 detail levels.

---

### 18. Flicker Reduction Optimizations

**pi-mono**: Single-line animation optimization, explicit flicker reduction for spinners.

**rollouts**: Standard differential rendering, no flicker-specific optimizations.

---

### 19. Full Redraw Tracking/Metrics

**pi-mono**: Tracks full redraw counts for performance monitoring.

**rollouts**: No render performance metrics.

---

### 20. Terminal Capability Queries

**pi-mono**: Cell dimension queries for images, Kitty keyboard protocol negotiation.

**rollouts**: Basic terminal detection only.

---

### 21. Key Hint Generation

**pi-mono**: `keyHint()` function generating contextual hints like `↑↓ navigate • enter select • esc cancel`.

**rollouts**: No dynamic keybinding hints.

---

### 22. Extensible Autocomplete Providers

**pi-mono**: Provider architecture with path, command, and custom providers.

**rollouts**: Single built-in completion mechanism.

---

### 23. Complete Escape Sequence Parsing

**pi-mono**: `StdinBuffer` with full CSI, OSC, DCS, APC, SS3 sequence detection.

**rollouts**: Basic stdin handling.

---

## Implementation Comparison

| Aspect | pi-mono | rollouts |
|--------|---------|----------|
| Language | TypeScript | Python |
| Base Library | `@mariozechner/pi-tui` | pytui (ported from pi-tui) |
| Core Components | 12 + 35 app-specific | 12 |
| Theme Colors | 51+ | ~15 |
| Overlay System | Full stack with anchors | None |
| Images | Kitty/iTerm2 | None |
| IME | Full support | None |
| Keybindings | 20+ configurable | Fixed |
| Kill Ring | Emacs-style | None |
| Undo | Full stack | None |
| Fuzzy Search | Scoring algorithm | Prefix only |
| Extensions | Widget/dialog API | None |
| Hyperlinks | OSC 8 | Text only |
| Syntax Highlighting | 9 token types | None |

---

## Priority Features

### High Impact

1. **Overlay/modal system** — Unlocks floating dialogs, confirmation modals, session browser
2. **Interactive session selector** — Currently have to remember session IDs
3. **Syntax highlighting** — Code blocks are hard to read without it
4. **Undo in editor** — Can't recover from mistakes

### Medium Impact

5. **Fuzzy search** — Better file/session discovery
6. **Kill ring** — Emacs users expect yank-pop
7. **Configurable keybindings** — Power users want customization
8. **OSC 8 hyperlinks** — Clickable links in supported terminals

### Lower Impact (Nice to Have)

9. **Image support** — Useful for vision models
10. **IME support** — Better CJK input
11. **Theme hot-reload** — Developer experience
12. **Render metrics** — Performance debugging

---

## Property-Based Testing Approach

To verify behavioral equivalence between pi-mono's TUI and rollouts' TUI, we could use property-based testing:

### Render Equivalence Properties

1. **Visible width calculation**: `visible_width(text) == pi_visible_width(text)` for arbitrary Unicode + ANSI
2. **Text wrapping**: `wrap(text, width)` produces same line breaks
3. **ANSI preservation**: Wrapped text preserves color codes at correct positions
4. **Truncation**: `truncate(text, width)` produces same visible output

### Component Behavior Properties

1. **Input handling**: Same keystrokes produce same cursor position and text state
2. **Selection navigation**: Up/down through N items wraps correctly
3. **Markdown rendering**: Same markdown produces equivalent ANSI output (modulo theme colors)

### Differential Rendering Properties

1. **Idempotence**: Rendering same state twice produces no terminal output
2. **Minimality**: Changed line count matches actual diff count
3. **Correctness**: Final terminal state matches full redraw

### Testing Strategy

- Use Hypothesis (Python) to generate arbitrary inputs
- Capture pi-mono behavior via subprocess with JSON output mode
- Compare against rollouts behavior
- Focus on shared primitives (text utils, components) rather than full TUI

---

## Glossary

### Kill Ring

The **kill ring** is an Emacs concept for clipboard history. Instead of a single clipboard:

- Every "kill" (delete) operation pushes deleted text onto a ring buffer
- **Yank** (Ctrl+Y) pastes the most recent kill
- **Yank-pop** (Alt+Y after Ctrl+Y) cycles through previous kills

Example workflow:
```
Delete "foo" (ring: [foo])
Delete "bar" (ring: [bar, foo])
Delete "baz" (ring: [baz, bar, foo])
Ctrl+Y → pastes "baz"
Alt+Y → replaces with "bar"
Alt+Y → replaces with "foo"
Alt+Y → cycles back to "baz"
```

pi-mono implements this with `killRing: string[]` and `killRingIndex`, supporting both character and word deletions that accumulate when consecutive.

### IME Support

**IME** (Input Method Editor) is how CJK languages (Chinese, Japanese, Korean) input characters:

1. User types phonetic keys (e.g., "nihon" for Japanese)
2. IME shows a **candidate window** with possible characters (日本, にほん, ニホン)
3. User selects the correct character

Terminal IME support requires:
- **Cursor position reporting**: The terminal must tell the IME where to draw the candidate window
- **Hardware cursor**: The blinking cursor must be at the actual input position
- **APC sequences**: Special escape codes (`ESC _ ... ESC \`) to communicate cursor position

Without IME support, CJK users see the candidate window in the wrong place or can't input at all.

pi-mono implements this via:
- `Focusable` interface that tracks which component has focus
- `CURSOR_MARKER` APC sequence embedded in render output
- Terminal shows hardware cursor at marker position
