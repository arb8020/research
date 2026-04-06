# pi-mono ↔ rollouts CLI: UI/UX gaps + reference snippets

Date: 2026-02-19  
pi-mono clone: `/tmp/pi-mono` @ `4ba3e5be229a570187d8efbef5c14c0d5ce40dcc`  
rollouts workspace: `/Users/chiraagbalu/research` @ `f531c4297cb3d8b20521d75a735c2bdf5657a561`

This doc focuses on “feel” gaps (keybindings, interrupt/exit semantics, expand/collapse affordances, input editor behavior, tool visibility/truncation), and includes the exact snippets used to ground each claim.

---

## 1) Keybinding system + discoverability (pi-mono) vs hardcoded keys (rollouts)

**Observed gap:** pi-mono documents a full action→keybinding mapping (and surfaces it via `/hotkeys`), while rollouts mostly hardcodes raw key sequences in the input loop and `Input.handle_input()`. This shows up as missing parity for common actions like:
- `ctrl+o` expand/collapse tool output
- `ctrl+t` hide/show thinking blocks
- `ctrl+l` model picker
- `alt+enter` follow-up queue
- `alt+up` dequeue

### pi-mono: keybinding docs enumerate actions + defaults

```md
/tmp/pi-mono/packages/coding-agent/docs/keybindings.md (excerpt, line-numbered)
    69	### Application
    70	
    71	| Action | Default | Description |
    72	|--------|---------|-------------|
    73	| `interrupt` | `escape` | Cancel / abort |
    74	| `clear` | `ctrl+c` | Clear editor |
    75	| `exit` | `ctrl+d` | Exit (when editor empty) |
    76	| `suspend` | `ctrl+z` | Suspend to background |
    77	| `externalEditor` | `ctrl+g` | Open in external editor (`$VISUAL` or `$EDITOR`) |
    78	
    79	### Session
    80	
    81	| Action | Default | Description |
    82	|--------|---------|-------------|
    83	| `newSession` | *(none)* | Start a new session (`/new`) |
    84	| `tree` | *(none)* | Open session tree navigator (`/tree`) |
    85	| `fork` | *(none)* | Fork current session (`/fork`) |
    86	| `resume` | *(none)* | Open session resume picker (`/resume`) |
    87	
    88	### Models & Thinking
    89	
    90	| Action | Default | Description |
    91	|--------|---------|-------------|
    92	| `selectModel` | `ctrl+l` | Open model selector |
    93	| `cycleModelForward` | `ctrl+p` | Cycle to next model |
    94	| `cycleModelBackward` | `shift+ctrl+p` | Cycle to previous model |
    95	| `cycleThinkingLevel` | `shift+tab` | Cycle thinking level |
    96	
    97	### Display
    98	
    99	| Action | Default | Description |
   100	|--------|---------|-------------|
   101	| `expandTools` | `ctrl+o` | Collapse/expand tool output |
   102	| `toggleThinking` | `ctrl+t` | Collapse/expand thinking blocks |
   103	
   104	### Message Queue
   105	
   106	| Action | Default | Description |
   107	|--------|---------|-------------|
   108	| `followUp` | `alt+enter` | Queue follow-up message |
   109	| `dequeue` | `alt+up` | Restore queued messages to editor |
```

### pi-mono: action handlers wired in interactive mode

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  1778		private setupKeyHandlers(): void {
  1779			// Set up handlers on defaultEditor - they use this.editor for text access
  1780			// so they work correctly regardless of which editor is active
  1781			this.defaultEditor.onEscape = () => {
  1782				if (this.loadingAnimation) {
  1783					this.restoreQueuedMessagesToEditor({ abort: true });
  1784				} else if (this.session.isBashRunning) {
  1785					this.session.abortBash();
  1786				} else if (this.isBashMode) {
  1787					this.editor.setText("");
  1788					this.isBashMode = false;
  1789					this.updateEditorBorderColor();
  1790				} else if (!this.editor.getText().trim()) {
  1791					// Double-escape with empty editor triggers /tree, /fork, or nothing based on setting
  1792					const action = this.settingsManager.getDoubleEscapeAction();
  1793					if (action !== "none") {
  1794						const now = Date.now();
  1795						if (now - this.lastEscapeTime < 500) {
  1796							if (action === "tree") {
  1797								this.showTreeSelector();
  1798							} else {
  1799								this.showUserMessageSelector();
  1800							}
  1801							this.lastEscapeTime = 0;
  1802						} else {
  1803							this.lastEscapeTime = now;
  1804						}
  1805					}
  1806				}
  1807			};
  1808	
  1809			// Register app action handlers
  1810			this.defaultEditor.onAction("clear", () => this.handleCtrlC());
  1811			this.defaultEditor.onCtrlD = () => this.handleCtrlD();
  1812			this.defaultEditor.onAction("suspend", () => this.handleCtrlZ());
  1813			this.defaultEditor.onAction("cycleThinkingLevel", () => this.cycleThinkingLevel());
  1814			this.defaultEditor.onAction("cycleModelForward", () => this.cycleModel("forward"));
  1815			this.defaultEditor.onAction("cycleModelBackward", () => this.cycleModel("backward"));
  1816	
  1817			// Global debug handler on TUI (works regardless of focus)
  1818			this.ui.onDebug = () => this.handleDebugCommand();
  1819			this.defaultEditor.onAction("selectModel", () => this.showModelSelector());
  1820			this.defaultEditor.onAction("expandTools", () => this.toggleToolOutputExpansion());
  1821			this.defaultEditor.onAction("toggleThinking", () => this.toggleThinkingBlockVisibility());
  1822			this.defaultEditor.onAction("externalEditor", () => this.openExternalEditor());
  1823			this.defaultEditor.onAction("followUp", () => this.handleFollowUp());
  1824			this.defaultEditor.onAction("dequeue", () => this.handleDequeue());
```

### rollouts: TUIFrontend reads raw keys (Ctrl+C / Esc) and routes everything else to focused component

```py
rollouts/rollouts/frontends/tui_frontend.py (excerpt, line-numbered)
   495	            while True:
   496	                if self._terminal and self._terminal._running:
   497	                    msg = self._terminal.read_message()
   498	                    if msg is not None:
   499	                        key = msg.key if isinstance(msg, KeyPress) else None
   500	                        # Check for Ctrl+C (ASCII 3) - double-tap to exit
   501	                        if key == "\x03":
   502	                            now = time.time()
   503	                            if (
   504	                                self._ctrl_c_pending
   505	                                and (now - self._ctrl_c_pending) < CTRL_C_TIMEOUT
   506	                            ):
   507	                                # Second Ctrl+C within timeout - send exit
   508	                                self._ctrl_c_pending = None
   509	                                if self._input_send:
   510	                                    try:
   511	                                        self._input_send.send_nowait(InputExit())
   512	                                    except trio.WouldBlock:
   513	                                        pass  # Channel full, exit will be handled
   514	                            else:
   515	                                # First Ctrl+C - show message and wait
   516	                                self._ctrl_c_pending = now
   517	                                if self._renderer:
   518	                                    self._renderer.add_system_message("Press Ctrl+C again to exit")
   519	                            continue
   520	
   521	                        # Check for Escape - interrupt current operation
   522	                        if key == "\x1b":
   523	                            # Set flag so get_input returns InputInterrupt
   524	                            self._interrupt_pending = True
   525	                            if self._renderer:
   526	                                self._renderer.add_system_message("Interrupted")
   527	                            # Send interrupt through channel to wake up get_input
   528	                            if self._input_send:
   529	                                try:
   530	                                    self._input_send.send_nowait(InputInterrupt())
   531	                                except trio.WouldBlock:
   532	                                    pass  # Channel full, interrupt flag is set
   533	                            continue
```

---

## 2) Ctrl+C semantics: “clear editor” (pi-mono) vs “exit flow” (rollouts)

**Observed gap:** pi-mono makes `ctrl+c` a first-class “clear editor” action; rollouts treats `ctrl+c` as “double-tap to exit” and doesn’t clear input.

### pi-mono: hotkeys explicitly describe “clear editor (first) / exit (second)”

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  4098	| \`${interrupt}\` | Cancel autocomplete / abort streaming |
  4099	| \`${clear}\` | Clear editor (first) / exit (second) |
  4100	| \`${exit}\` | Exit (when editor is empty) |
```

### rollouts: Ctrl+C is reserved at the frontend input loop level

```py
rollouts/rollouts/frontends/tui_frontend.py (excerpt, line-numbered)
   500	                        # Check for Ctrl+C (ASCII 3) - double-tap to exit
   501	                        if key == "\x03":
   502	                            now = time.time()
   503	                            if (
   504	                                self._ctrl_c_pending
   505	                                and (now - self._ctrl_c_pending) < CTRL_C_TIMEOUT
   506	                            ):
   507	                                # Second Ctrl+C within timeout - send exit
   508	                                ...
   514	                            else:
   515	                                # First Ctrl+C - show message and wait
   516	                                self._ctrl_c_pending = now
   517	                                if self._renderer:
   518	                                    self._renderer.add_system_message("Press Ctrl+C again to exit")
   519	                            continue
```

---

## 3) Escape semantics: contextual cancel/abort (pi-mono) vs unconditional interrupt (rollouts)

**Observed gap:** pi-mono’s `Esc` handler is stateful (abort queued messages, abort bash, exit bash-mode, double-escape actions when editor empty). rollouts turns any `Esc` into a runner interrupt signal.

### pi-mono: `onEscape` is contextual and supports “double escape” actions

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  1781			this.defaultEditor.onEscape = () => {
  1782				if (this.loadingAnimation) {
  1783					this.restoreQueuedMessagesToEditor({ abort: true });
  1784				} else if (this.session.isBashRunning) {
  1785					this.session.abortBash();
  1786				} else if (this.isBashMode) {
  1787					this.editor.setText("");
  1788					this.isBashMode = false;
  1789					this.updateEditorBorderColor();
  1790				} else if (!this.editor.getText().trim()) {
  1791					// Double-escape with empty editor triggers /tree, /fork, or nothing based on setting
  1792					...
  1795						if (now - this.lastEscapeTime < 500) {
  1796							if (action === "tree") {
  1797								this.showTreeSelector();
  1798							} else {
  1799								this.showUserMessageSelector();
  1800							}
  1801							this.lastEscapeTime = 0;
  1802						} else {
  1803							this.lastEscapeTime = now;
  1804						}
```

### rollouts: `Esc` becomes `InputInterrupt` and the runner translates it into `StopReason.INTERRUPTED`

```py
rollouts/rollouts/frontends/runner.py (excerpt, line-numbered)
   503	            match input_result:
   504	                case InputExit():
   505	                    return dc_replace(state, stop=StopReason.NO_TOOL_CALLED)
   506	
   507	                case InputInterrupt():
   508	                    # User pressed Escape - signal interrupted so main loop handles it
   509	                    return dc_replace(state, stop=StopReason.INTERRUPTED)
```

---

## 4) Tool output expansion: pi-mono has a global toggle; rollouts has the plumbing but no toggle

**Observed gap:** rollouts already supports different tool display modes (`compact|standard|expanded`) and formatting honors it, but there’s no keybinding to cycle/toggle it the way pi-mono does (`ctrl+o`).

### pi-mono: global “set expanded” walks expandable chat children

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  2702		private toggleToolOutputExpansion(): void {
  2703			this.setToolsExpanded(!this.toolOutputExpanded);
  2704		}
  2705	
  2706		private setToolsExpanded(expanded: boolean): void {
  2707			this.toolOutputExpanded = expanded;
  2708			for (const child of this.chatContainer.children) {
  2709				if (isExpandable(child)) {
  2710					child.setExpanded(expanded);
  2711				}
  2712			}
  2713			this.ui.requestRender();
  2714		}
```

### rollouts: theme exposes `tool_display` modes

```py
pytui/pytui/theme.py (excerpt, line-numbered)
    12	# Tool display modes
    13	ToolDisplayMode = Literal["compact", "standard", "expanded"]
    ...
    98	    # Tool display mode: compact (one-liner), standard (truncated), expanded (full)
    99	    tool_display: ToolDisplayMode = "standard"
   100	
   101	    # Lines shown in standard mode (expanded = unlimited, compact = 0)
   102	    tool_lines_standard: int = 10
```

### rollouts: formatting uses `theme.tool_display` (but no key handler changes it)

```py
rollouts/rollouts/environments/_formatting.py (excerpt, line-numbered)
   233	def format_tool_themed(
   234	    tool_name: str,
   235	    args: dict[str, Any],
   236	    result: dict[str, Any] | None,
   237	    theme: Theme | None = None,
   238	    config: ToolRenderConfig | None = None,
   239	) -> str:
   260	    # Get display mode from theme (default to standard)
   261	    display_mode = "standard"
   262	    if theme and hasattr(theme, "tool_display"):
   263	        display_mode = theme.tool_display
   264	    ...
   285	    if display_mode == "compact":
   286	        if result is None:
   287	            # Pending - no indicator yet
   288	            return header
   289	        is_error = result.get("isError", False)
   290	        indicator = "✗" if is_error else "✓"
   291	        return f"{header} {indicator}"
```

### rollouts: ToolExecution rebuild uses formatter each time; it’s ready for a global toggle

```py
rollouts/rollouts/frontends/tui/components/tool_execution.py (excerpt, line-numbered)
    90	    def set_expanded(self, expanded: bool) -> None:
    91	        """Set whether to show expanded output (legacy, prefer set_detail_level)."""
    92	        from ....dtypes import DetailLevel
    93	
    94	        self._detail_level = DetailLevel.EXPANDED if expanded else DetailLevel.STANDARD
    95	        self._rebuild_display()
    ...
   188	    def _format_tool_execution(self) -> str:
   189	        """Format tool execution display.
   190	
   191	        Uses theme's tool_display mode (compact/standard/expanded) for formatting.
   192	        Falls back to legacy formatting if no theme available.
   193	        """
   194	        from ....environments._formatting import format_tool_themed
   195	        return format_tool_themed(
   196	            self._tool_name,
   197	            self._args,
   198	            self._result,
   199	            self._theme,
   200	            self._render_config,
   201	        )
```

---

## 5) Thinking blocks toggle: pi-mono can hide/show; rollouts only has “compact render” behavior

**Observed gap:** pi-mono has `ctrl+t` to hide/show thinking blocks and rebuilds chat. rollouts renders thinking blocks and can *render them compactly* when tool display mode is compact, but there’s no user-facing toggle.

### pi-mono: toggling thinking rebuilds chat + persists setting

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  2716		private toggleThinkingBlockVisibility(): void {
  2717			this.hideThinkingBlock = !this.hideThinkingBlock;
  2718			this.settingsManager.setHideThinkingBlock(this.hideThinkingBlock);
  2719	
  2720			// Rebuild chat from session messages
  2721			this.chatContainer.clear();
  2722			this.rebuildChatFromMessages();
  2723	
  2724			// If streaming, re-add the streaming component with updated visibility and re-render
  2725			if (this.streamingComponent && this.streamingMessage) {
  2726				this.streamingComponent.setHideThinkingBlock(this.hideThinkingBlock);
  2727				this.streamingComponent.updateContent(this.streamingMessage);
  2728				this.chatContainer.addChild(this.streamingComponent);
  2729			}
  2730	
  2731			this.showStatus(`Thinking blocks: ${this.hideThinkingBlock ? "hidden" : "visible"}`);
  2732		}
```

### rollouts: thinking is always rendered; “compact” mode just changes formatting

```py
rollouts/rollouts/frontends/tui/components/assistant_message.py (excerpt, line-numbered)
    56	    def append_thinking(self, delta: str) -> None:
    57	        """Append thinking delta to current thinking content."""
    58	        self._thinking_content += delta
    59	        if self._thinking_md:
    60	            # Update existing component
    61	            is_compact = (
    62	                hasattr(self._theme, "tool_display") and self._theme.tool_display == "compact"
    63	            )
    64	            if is_compact:
    65	                # Compact mode: one-liner, truncated
    66	                content = self._thinking_content.strip().replace("\n", " ")
    67	                if len(content) > 60:
    68	                    content = content[:57] + "..."
    69	                thinking_text = f"thinking() {content}"
    70	            else:
    71	                thinking_text = f"thinking()\n\n{self._thinking_content.strip()}"
    72	            self._thinking_md.set_text(thinking_text)
    ...
   123	        # Render thinking blocks first (if any)
   124	        if self._thinking_content and self._thinking_content.strip():
   125	            if is_compact:
   126	                ...
   131	                thinking_text = f"thinking() {content}"
   132	                bg_fn = lambda x: x  # No background in compact mode  # noqa: E731
   133	                padding_y = 0
   134	            else:
   135	                thinking_text = f"thinking()\n\n{self._thinking_content.strip()}"
```

---

## 6) Multiline compose: pi-mono has an explicit “new line” action; rollouts submits on Enter

**Observed gap:** pi-mono keybinding model includes `newLine` (default `shift+enter`) distinct from submit; rollouts `Input` currently submits on carriage return and doesn’t implement a “new line” key path.

### pi-mono: `/hotkeys` table includes explicit “New line”

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  4043			// Editing keybindings
  4044			const submit = this.getEditorKeyDisplay("submit");
  4045			const newLine = this.getEditorKeyDisplay("newLine");
  ...
  4084	| \`${submit}\` | Send message |
  4085	| \`${newLine}\` | New line${process.platform === "win32" ? " (Ctrl+Enter on Windows Terminal)" : ""} |
```

### rollouts: Enter submits (CR) and resets editor

```py
rollouts/rollouts/frontends/tui/components/input.py (excerpt, line-numbered)
   240	        # Enter - submit
   241	        if len(data) == 1 and ord(data[0]) == 13:  # CR
   242	            if self._disable_submit:
   243	                return
   244	
   245	            result = self.get_text().strip()
   246	            # Replace paste markers
   247	            for paste_id, paste_content in self._pastes.items():
   248	                ...
   253	            # Reset editor
   254	            self._lines = [""]
   255	            self._cursor_line = 0
   256	            self._cursor_col = 0
   257	            self._pastes.clear()
   258	            self._paste_counter = 0
   259	
   260	            if self._on_change:
   261	                self._on_change("")
   262	
   263	            if self._on_submit:
   264	                self._on_submit(result)
   265	            return
```

---

## 7) Tab completion / “@file references”: pi-mono supports both; rollouts has stubs but they’re unwired

**Observed gap:** rollouts `Input` supports Tab-completion callbacks + ghost text preview, but `TUIFrontend.start()` never wires a completion callback. pi-mono supports `@file` arguments and “Tab completion / accept autocomplete” as a core UX affordance.

### pi-mono: CLI supports `@file` arguments

```ts
/tmp/pi-mono/packages/coding-agent/src/cli/args.ts (excerpt, line-numbered)
   152			} else if (arg === "--verbose") {
   153				result.verbose = true;
   154			} else if (arg.startsWith("@")) {
   155				result.fileArgs.push(arg.slice(1)); // Remove @ prefix
   156			} else if (arg.startsWith("--") && extensionFlags) {
```

### pi-mono: `/hotkeys` mentions Tab completion

```ts
/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts (excerpt, line-numbered)
  4097	| \`${tab}\` | Path completion / accept autocomplete |
```

### rollouts: Input implements Tab completion if a callback exists

```py
rollouts/rollouts/frontends/tui/components/input.py (excerpt, line-numbered)
   282	        # Tab - trigger completion
   283	        if len(data) == 1 and ord(data[0]) == 9:
   284	            self._handle_tab_complete()
   285	            return
   ...
   350	    def _handle_tab_complete(self) -> None:
   351	        """Handle Tab key for completion."""
   352	        if not self._on_tab_complete:
   353	            return
   354	
   355	        text = self.get_text()
   356	        completed = self._on_tab_complete(text)
```

### rollouts: TUIFrontend creates Input but does not set `set_on_tab_complete(...)`

```py
rollouts/rollouts/frontends/tui_frontend.py (excerpt, line-numbered)
   115	        # Create input component
   116	        self._input_component = Input(theme=self._tui.theme)
   117	        self._input_component.set_on_submit(self._handle_input_submit)
   118	        self._input_component.set_on_editor(self._handle_open_editor)
```

---

## 8) Message queue UX: pi-mono has steering/follow-up concepts; rollouts shows a simple queue list

**Observed gap:** rollouts does show queued messages near the editor and supports restoring via Up-arrow, but it doesn’t expose pi-mono-like follow-up vs steering semantics or the keybindings in docs (`alt+enter`, `alt+up`).

### pi-mono: queue-related actions are explicitly part of the keybinding model

```md
/tmp/pi-mono/packages/coding-agent/docs/keybindings.md (excerpt, line-numbered)
   104	### Message Queue
   105	
   106	| Action | Default | Description |
   107	|--------|---------|-------------|
   108	| `followUp` | `alt+enter` | Queue follow-up message |
   109	| `dequeue` | `alt+up` | Restore queued messages to editor |
```

### rollouts: queued messages are displayed via PendingMessages (near-editor component)

```py
rollouts/rollouts/frontends/tui/components/pending_messages.py (excerpt, line-numbered)
    12	class PendingMessages(Component):
    13	    """Shows queued messages and a hint to restore them into the editor."""
    ...
    56	        reason = self._get_blocked_reason() if self._get_blocked_reason else None
    57	        if reason:
    58	            header = f"{len(self._messages)} queued. Blocked: {reason}. {self._restore_hint}"
    59	        else:
    60	            header = f"{len(self._messages)} queued. {self._restore_hint}"
```

---

## 9) Tool surface area: pi-mono exposes `grep/find/ls`; rollouts “coding” env does not

**Observed gap:** pi-mono’s CLI help advertises more built-in tools and includes them in `--tools`. rollouts’ LocalFilesystemEnvironment tools are currently `read/write/edit/bash/web_fetch` (plus other environments elsewhere, but not wired as “coding tools”).

### pi-mono: CLI help lists available tools (includes grep/find/ls)

```ts
/tmp/pi-mono/packages/coding-agent/src/cli/args.ts (excerpt, line-numbered)
   205	  --no-tools                     Disable all built-in tools
   206	  --tools <tools>                Comma-separated list of tools to enable (default: read,bash,edit,write)
   207	                                 Available: read, bash, edit, write, grep, find, ls
```

### rollouts: coding environment tool list is fixed in `_get_all_tools()`

```py
rollouts/rollouts/environments/coding.py (excerpt, line-numbered)
   645	    def _get_all_tools(self) -> list[Tool]:
   646	        """Return all available tools (before filtering)."""
   647	        return [
   648	            # read tool
   649	            Tool(
   650	                type="function",
   651	                function=ToolFunction(
   652	                    name="read",
   653	                    ...
   674	            # write tool
   675	            Tool(
   676	                type="function",
   677	                function=ToolFunction(
   678	                    name="write",
   679	                    ...
   696	            # edit tool
   697	            Tool(
   698	                type="function",
   699	                function=ToolFunction(
   700	                    name="edit",
   701	                    ...
   722	            # bash tool
   723	            Tool(
   724	                type="function",
   725	                function=ToolFunction(
   726	                    name="bash",
   727	                    ...
   741	            # web_fetch tool
   742	            Tool(
   743	                type="function",
   744	                function=ToolFunction(
   745	                    name="web_fetch",
   746	                    ...
   763	        ]
```

---

## Notes / quick “fix targets” this evidence suggests

- Add an app-level keybinding layer in rollouts `TUIFrontend.run_input_loop()` so `ctrl+o` and `ctrl+t` can toggle `theme.tool_display` and a “hide thinking” flag without leaking those keys to `Input.handle_input()`.
- Reconcile `ctrl+c` semantics with pi-mono: clear editor first; keep double-`ctrl+c` to exit.
- Make `Esc` contextual: if not streaming / not blocked, treat as “cancel UI/autocomplete” rather than emitting `InputInterrupt` unconditionally.
- Wire `Input.set_on_tab_complete(...)` (and potentially a file index) to unlock Tab completion and match pi-mono’s baseline UX.

