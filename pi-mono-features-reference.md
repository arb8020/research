# pi-mono Features Reference

Quick reference for features we want to adopt from badlogic/pi-mono, with exact file locations.

---

## 1. Diff Rendering for Edit Operations ⭐ PRIORITY

**What it does**: Shows colored diff output with +/- for added/removed lines when editing files

**Where to find it**:

### Edit Tool Implementation
- **File**: `pi-mono/packages/coding-agent/src/tools/edit.ts`
- **Key logic**: Lines 40-80 - Generates unified diff with line numbers and context
- **Returns**: `details: { diff: string }` in the result

### Diff Rendering in TUI
- **File**: `pi-mono/packages/coding-agent/src/tui/tool-execution.ts`
- **Key logic**: Lines 180-220 - `formatEdit()` function
- **Color tokens used**:
  - `theme.toolDiffAdded` - Green for added lines (lines starting with +)
  - `theme.toolDiffRemoved` - Red for removed lines (lines starting with -)
  - `theme.toolDiffContext` - Muted for context lines (lines starting with space)

### Theme Colors
- **File**: `pi-mono/packages/coding-agent/src/theme/dark.json`
- **Tokens**:
  ```json
  "toolDiffAdded": "#2d4a2d",
  "toolDiffRemoved": "#4a2d2d",
  "toolDiffContext": "#32323c"
  ```

---

## 2. Additional Tools (grep/find/ls)

### Grep Tool
**What it does**: Search file contents using ripgrep with regex/literal patterns, glob filters, context lines

**Where to find it**:
- **Implementation**: `pi-mono/packages/coding-agent/src/tools/grep.ts`
- **Auto-download logic**: `pi-mono/packages/coding-agent/src/tools-manager.ts` lines 80-150
- **Key features**:
  - Pattern (required), path, glob, ignoreCase, literal, context, limit
  - Default limit: 100 matches
  - JSON output parsing for precise line numbers
  - Respects .gitignore

### Find Tool
**What it does**: Fast file finding using fd with glob patterns

**Where to find it**:
- **Implementation**: `pi-mono/packages/coding-agent/src/tools/find.ts`
- **Auto-download logic**: `pi-mono/packages/coding-agent/src/tools-manager.ts` lines 150-220
- **Key features**:
  - Glob pattern (required), path, limit
  - Default limit: 1000 results
  - Respects .gitignore

### Ls Tool
**What it does**: List directory contents with optional limit

**Where to find it**:
- **Implementation**: `pi-mono/packages/coding-agent/src/tools/ls.ts`
- **Key features**:
  - Optional path and limit
  - Default limit: 500 entries
  - Directory suffix (`/`)
  - Alphabetical sorting

---

## 3. Tilde Expansion in Paths

**What it does**: Converts `~/` to user home directory in all path parameters

**Where to find it**:
- **Utility function**: `pi-mono/packages/coding-agent/src/tools/path-utils.ts`
- **Function**: `expandTilde(path: string): string`
- **Used in**: All tools (read.ts, write.ts, edit.ts, ls.ts, etc.)
- **Example**:
  ```typescript
  const expandedPath = expandTilde(path); // ~/code -> /Users/username/code
  ```

---

## 4. Tool-Specific Formatters with Better Truncation

**What it does**: Different max line limits per tool, smart truncation with "..." indicators

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tui/tool-execution.ts`
- **Tool-specific limits**:
  - `formatBash()` - 5 lines max (line 120)
  - `formatRead()` - 10 lines max (line 150)
  - `formatWrite()` - 10 lines max (line 170)
  - `formatEdit()` - Shows full diff (line 190)
  - `formatLs/Find/Grep()` - Varies by tool

**Helper function**:
- `truncateLines(text: string, maxLines: number, showContinuation: boolean)` (line 250)

---

## 5. Path Shortening in Displays

**What it does**: Shows `~/file.ts:10-100` instead of full absolute paths

**Where to find it**:
- **Utility function**: `pi-mono/packages/coding-agent/src/tools/path-utils.ts`
- **Function**: `shortenPath(path: string, homeDir: string): string`
- **Used in**: Tool formatters (tool-execution.ts)
- **Example**:
  ```typescript
  // /Users/username/code/file.ts -> ~/code/file.ts
  const short = shortenPath(fullPath, os.homedir());
  ```

**Line range formatting**:
- **File**: `pi-mono/packages/coding-agent/src/tui/tool-execution.ts`
- **Logic**: Lines 150-160 in `formatRead()`
- **Example**: `~/file.ts:10-100` (shows offset and limit)

---

## 6. Image Support in Read Tool

**What it does**: Detects image files and returns base64-encoded data with MIME type

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tools/read.ts`
- **Logic**: Lines 20-40
- **Supported formats**: jpg, jpeg, png, gif, webp
- **Returns**: `{ isImage: true, mimeType: 'image/png', data: 'base64...' }`

**Rendering**:
- **File**: `pi-mono/packages/coding-agent/src/tui/tool-execution.ts`
- **Logic**: Lines 150-155 in `formatRead()`
- **Display**: `[Image: image/png]` instead of binary data

---

## 7. Tab Replacement in Output

**What it does**: Converts tabs to 3 spaces for cleaner, more consistent display

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tui/tool-execution.ts`
- **Function**: `replaceTab(text: string): string` (line 270)
- **Usage**: Applied to all tool output before display
- **Example**:
  ```typescript
  const cleaned = replaceTab(output); // \t -> "   " (3 spaces)
  ```

---

## 8. Enhanced Theme System (44 Color Tokens)

**What it does**: Comprehensive color system with variables for reuse

**Where to find it**:
- **Interface**: `pi-mono/packages/coding-agent/src/theme/theme.ts`
- **Default themes**:
  - `pi-mono/packages/coding-agent/src/theme/dark.json`
  - `pi-mono/packages/coding-agent/src/theme/light.json`

**Color categories**:
1. **Core UI** (10): accent, border, success, error, warning, muted, dim, text, etc.
2. **Backgrounds** (7): userMessageBg, toolPendingBg, toolSuccessBg, toolErrorBg, etc.
3. **Markdown** (10): mdHeading, mdLink, mdCode, mdCodeBlock, mdQuote, etc.
4. **Tool Diffs** (3): toolDiffAdded, toolDiffRemoved, toolDiffContext
5. **Syntax** (9): syntaxComment, syntaxKeyword, syntaxFunction, syntaxString, etc.
6. **Thinking** (5): thinkingOff, thinkingMinimal, thinkingLow, thinkingMedium, thinkingHigh

**Variable system**:
```json
{
  "variables": {
    "base": "#1e1e2e"
  },
  "colors": {
    "userMessageBg": "$base"
  }
}
```

---

## 9. Timeout Parameter in Bash Tool

**What it does**: Optional timeout (in seconds) for bash commands with automatic termination

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tools/bash.ts`
- **Parameter**: `timeout?: number` (line 15)
- **Logic**: Lines 50-70 - Uses AbortSignal with setTimeout
- **Default**: No timeout (runs until completion)

**Process cleanup**:
- **Tree killing**: Lines 80-100 - Kills entire process tree on abort/timeout
- **Platform-specific**: Different kill methods for Unix/Windows

---

## 10. Offset/Limit Pagination in Read Tool

**What it does**: Read large files in chunks with offset (starting line) and limit (max lines)

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tools/read.ts`
- **Parameters**:
  - `offset?: number` - Starting line number (0-indexed)
  - `limit?: number` - Max lines to read (default: 2000)
- **Logic**: Lines 45-65
- **Example**:
  ```typescript
  // Read lines 100-200 from file
  read({ path: "~/file.ts", offset: 100, limit: 100 })
  ```

**Display in TUI**:
- **Shows range**: `~/file.ts:100-200` (200 lines)

---

## 11. Auto-Download System (ripgrep/fd)

**What it does**: Automatically downloads ripgrep and fd binaries from GitHub releases if not found

**Where to find it**:
- **File**: `pi-mono/packages/coding-agent/src/tools-manager.ts`
- **Functions**:
  - `ensureRipgrep()` - Lines 80-150
  - `ensureFd()` - Lines 150-220
  - `downloadTool()` - Lines 30-80 (generic downloader)

**Features**:
- Platform detection (darwin, linux, win32)
- GitHub API for latest releases
- Progress indicators
- Checksum validation
- Extract to `~/.pi/bin/`
- Fallback to system PATH

**Usage**:
```typescript
const rgPath = await ensureRipgrep(); // Returns path to rg binary
```

---

## 12. Abort Signal Support

**What it does**: All tools accept AbortSignal for graceful cancellation

**Where to find it**:
- **Tool signature**: Every tool accepts `signal?: AbortSignal` parameter
- **Example**: `pi-mono/packages/coding-agent/src/tools/bash.ts` lines 60-80
- **Features**:
  - Process tree killing for bash
  - Stream cleanup for read/write
  - Early return on abort
  - No partial writes

---

## Implementation Priority

Based on impact and complexity:

1. ⭐ **Diff Rendering** - High impact, medium complexity
2. **Tool-Specific Formatters** - High impact, low complexity
3. **Tilde Expansion** - Medium impact, low complexity
4. **Path Shortening** - Medium impact, low complexity
5. **grep/find/ls Tools** - High impact, high complexity (auto-download)
6. **Tab Replacement** - Low impact, low complexity
7. **Enhanced Theme System** - Medium impact, medium complexity
8. **Timeout in Bash** - Low impact, low complexity
9. **Offset/Limit in Read** - Medium impact, medium complexity
10. **Image Support** - Low impact, medium complexity
11. **Auto-Download System** - Medium impact, high complexity
12. **Abort Signal** - Low impact (already have?), medium complexity

---

## Notes

- All file paths are relative to `/Users/chiraagbalu/research/pi-mono/`
- TypeScript source files are in `packages/coding-agent/src/` and `packages/web-ui/src/`
- Focus on coding-agent package (TUI/CLI), not web-ui (browser-based)
