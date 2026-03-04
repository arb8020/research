/**
 * Claude/Codex CLI Provider Extension
 *
 * Provides stream-native Pi providers backed by local CLI binaries:
 * - claude    -> `claude --output-format stream-json`
 * - codex     -> `codex exec --json`
 */

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import type { Context, Message } from "@mariozechner/pi-ai";
import {
	calculateCost,
	createAssistantMessageEventStream,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	type Api,
	type Model,
	type SimpleStreamOptions,
	type StopReason,
} from "@mariozechner/pi-ai";
import { bashTool, readTool, writeTool, editTool, grepTool, findTool } from "@mariozechner/pi-coding-agent";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const CLAUDE_CLI_PROVIDER_API = "claude-cli-json";
const CODEX_CLI_PROVIDER_API = "codex-cli-json";

const PROVIDER_API_KEY_PLACEHOLDER = "PI_CLI_PROVIDER_API_KEY";

interface CliModelDefinition {
	id: string;
	name: string;
	apiModel: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
	};
	contextWindow: number;
	maxTokens: number;
}

const CLAUDE_CLI_MODELS: CliModelDefinition[] = [
	// Aliases (resolved by the claude CLI itself)
	{
		id: "sonnet",
		apiModel: "sonnet",
		name: "Claude Sonnet (CLI, latest)",
		reasoning: true,
		input: ["text"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	},
	{
		id: "opus",
		apiModel: "opus",
		name: "Claude Opus (CLI, latest)",
		reasoning: true,
		input: ["text"],
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 200_000,
		maxTokens: 32_000,
	},
	// Pinned versions
	{
		id: "claude-sonnet-4-6",
		apiModel: "claude-sonnet-4-6",
		name: "Claude Sonnet 4.6 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	},
	{
		id: "claude-sonnet-4-5-20250929",
		apiModel: "claude-sonnet-4-5-20250929",
		name: "Claude Sonnet 4.5 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	},
	{
		id: "claude-opus-4-5-20251101",
		apiModel: "claude-opus-4-5-20251101",
		name: "Claude Opus 4.5 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	},
	{
		id: "claude-opus-4-1-20250805",
		apiModel: "claude-opus-4-1-20250805",
		name: "Claude Opus 4.1 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 15, output: 75, cacheRead: 1.5, cacheWrite: 18.75 },
		contextWindow: 200_000,
		maxTokens: 32_000,
	},
	{
		id: "claude-haiku-4-5-20251001",
		apiModel: "claude-haiku-4-5-20251001",
		name: "Claude Haiku 4.5 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
		contextWindow: 200_000,
		maxTokens: 64_000,
	},
];

const CODEX_CLI_MODELS: CliModelDefinition[] = [
	{
		id: "gpt-5",
		apiModel: "gpt-5",
		name: "GPT-5 (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 2.5, output: 10, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 16_000,
	},
	{
		id: "o3-mini",
		apiModel: "o3-mini",
		name: "O3 Mini (CLI)",
		reasoning: true,
		input: ["text"],
		cost: { input: 1, output: 4, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 16_000,
	},
];

const CLAUDE_MODELS_BY_ID = new Map(CLAUDE_CLI_MODELS.map((m) => [m.id, m]));
const CODEX_MODELS_BY_ID = new Map(CODEX_CLI_MODELS.map((m) => [m.id, m]));

function createUsageTemplate() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createAssistantTemplate(model: Model<Api>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: createUsageTemplate(),
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function toNumber(value: unknown): number {
	if (typeof value === "number" && Number.isFinite(value)) return value;
	if (typeof value === "string") {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : 0;
	}
	return 0;
}

function toIndex(value: unknown): number | undefined {
	if (typeof value === "number" && Number.isInteger(value)) return value;
	if (typeof value === "string") {
		const parsed = Number(value);
		return Number.isInteger(parsed) ? parsed : undefined;
	}
	return undefined;
}

function blockToText(block: unknown): string {
	if (typeof block === "string") return block;
	if (!block || typeof block !== "object") return String(block ?? "");
	const typed = block as Record<string, unknown>;
	switch (typed.type) {
		case "text": {
			const text = typed.text;
			return typeof text === "string" ? text : "";
		}
		case "thinking": {
			const text = typed.thinking;
			return typeof text === "string" ? text : "";
		}
		case "toolCall": {
			const name = typeof typed.name === "string" ? typed.name : "tool";
			const args = typed.arguments;
			const rendered = args ? JSON.stringify(args) : "{}";
			return `ToolCall ${name}: ${rendered}`;
		}
		case "image":
			return "[image omitted]";
		default:
			return JSON.stringify(block);
	}
}

function messageContentToText(content: Message["content"]): string {
	if (typeof content === "string") {
		return content.trim();
	}
	return content
		.map((block) => {
			if (typeof block === "string") return block;
			if (block?.type === "text") return block.text;
			if (block?.type === "thinking") return block.thinking;
			if (block?.type === "image") return "[image omitted]";
			if (block?.type === "toolCall") {
				return `tool call: ${block.name} ${JSON.stringify(block.arguments || {})}`;
			}
			return blockToText(block);
		})
		.filter(Boolean)
		.join("\n");
}

function formatContextPrompt(context: Context): string {
	const lines: string[] = [];

	if (context.systemPrompt?.trim()) {
		lines.push(`System prompt:\n${context.systemPrompt.trim()}`);
	}

	for (const msg of context.messages.slice(-24)) {
		if (msg.role === "toolResult") {
			const content = msg.content
				.map((c) => (c.type === "text" ? c.text : "[image]"))
				.join("\n")
				.trim();
			const status = msg.isError ? " (error)" : "";
			lines.push(`TOOL RESULT [${msg.toolName}]${status}:\n${content || "(no output)"}`);
			continue;
		}
		const contentText = messageContentToText(msg.content);
		if (!contentText) continue;
		lines.push(`${msg.role.toUpperCase()}:\n${contentText}`);
	}

	return lines.join("\n\n").trim();
}

function mapClaudeStopReason(subtype?: string): Extract<StopReason, "stop" | "length" | "toolUse"> {
	if (!subtype) return "stop";
	switch (subtype) {
		case "max_tokens":
			return "length";
		case "tool_use":
			return "toolUse";
		default:
			return "stop";
	}
}

function applyUsage(
	target: AssistantMessage["usage"],
	rawUsage: Record<string, unknown> | undefined,
): void {
	target.input = toNumber(rawUsage?.input_tokens ?? rawUsage?.input);
	target.output = toNumber(rawUsage?.output_tokens ?? rawUsage?.output);
	target.cacheRead = toNumber(rawUsage?.cache_read_input_tokens);
	target.cacheWrite = toNumber(rawUsage?.cache_creation_input_tokens);
	target.totalTokens = target.input + target.output + target.cacheRead + target.cacheWrite;
}

class ClaudeCliParser {
	private output: AssistantMessage;
	private readonly model: Model<Api>;
	private finished = false;
	private errored = false;
	private streamedContent = false;

	private readonly textByOutputIndex = new Map<number, string>();
	private readonly thinkingByOutputIndex = new Map<number, string>();
	private readonly toolsByOutputIndex = new Map<
		number,
		{ id: string; name: string; argsJson: string; contentIndex: number }
	>();
	private readonly streamBlockIndexToOutputIndex = new Map<number, number>();

	constructor(model: Model<Api>) {
		this.output = createAssistantTemplate(model);
		this.model = model;
	}

	get isFinished(): boolean {
		return this.finished || this.errored;
	}

	get message(): AssistantMessage {
		return this.output;
	}

	get finalResult(): AssistantMessage {
		return this.output;
	}

	parseLine(msg: Record<string, unknown>): AssistantMessageEvent[] {
		const events: AssistantMessageEvent[] = [];

		if (this.isFinished) {
			return events;
		}

		const msgType = msg.type;
		if (msgType === "system" && msg.subtype === "init") {
			// System init carries metadata. CLI streaming is already in progress after we emit start externally.
			return events;
		}

		if (msgType === "assistant") {
			if (this.streamedContent) return events;
			const message = (msg.message as Record<string, unknown> | undefined) ?? {};
			const content = Array.isArray(message.content) ? message.content : [];
			for (let i = 0; i < content.length; i += 1) {
				events.push(...this.parseCompleteBlock(content[i]));
			}
		} else if (msgType === "stream_event") {
			const event = (msg.event as Record<string, unknown>) || {};
			events.push(...this.parseStreamEvent(event));
		} else if (msgType === "result") {
			const isError = !!msg.is_error;
			const usage = msg.usage as Record<string, unknown> | undefined;
			if (isError) {
				this.errored = true;
				this.output.stopReason = "error";
				this.output.errorMessage = String((msg as Record<string, unknown>).result ?? "Unknown error");
				events.push({ type: "error", reason: "error", error: this.output });
			} else {
				applyUsage(this.output.usage, usage);
				calculateCost(this.model, this.output.usage);
				this.output.stopReason = mapClaudeStopReason((msg.subtype as string | undefined) ?? undefined);
				this.finished = true;
				events.push({ type: "done", reason: this.output.stopReason, message: this.output });
			}
		}

		return events;
	}

	private parseStreamEvent(event: Record<string, unknown>): AssistantMessageEvent[] {
		const events: AssistantMessageEvent[] = [];
		const eventType = event.type;
		const index = toIndex(event.index);

		switch (eventType) {
			case "message_start": {
				break;
			}
			case "content_block_start": {
				const block = (event.content_block as Record<string, unknown>) || {};
				const blockType = String(block.type ?? "");
				if (typeof index !== "number") break;
				this.streamedContent = true;
				const outputIndex = this.output.content.length;
				this.streamBlockIndexToOutputIndex.set(index, outputIndex);

				if (blockType === "text") {
					this.output.content.push({ type: "text", text: "" });
					this.textByOutputIndex.set(outputIndex, "");
					events.push({ type: "text_start", contentIndex: outputIndex, partial: this.output });
				} else if (blockType === "thinking") {
					this.output.content.push({ type: "thinking", thinking: "" });
					this.thinkingByOutputIndex.set(outputIndex, "");
					events.push({ type: "thinking_start", contentIndex: outputIndex, partial: this.output });
				} else if (blockType === "tool_use") {
					const toolId = String(block.id ?? "");
					const toolName = String(block.name ?? "tool");
					this.output.content.push({
						type: "toolCall",
						id: toolId,
						name: toolName,
						arguments: {},
					});
					this.toolsByOutputIndex.set(outputIndex, { id: toolId, name: toolName, argsJson: "", contentIndex: outputIndex });
					events.push({ type: "toolcall_start", contentIndex: outputIndex, partial: this.output });
				}
				break;
			}
			case "content_block_delta": {
				const delta = (event.delta as Record<string, unknown>) || {};
				const outputIndex = this.streamBlockIndexToOutputIndex.get(index);
				const deltaType = String(delta.type ?? "");
				if (typeof outputIndex !== "number" || typeof index !== "number") break;

				if (deltaType === "text_delta" && typeof outputIndex === "number") {
					const text = String(delta.text ?? "");
					this.textByOutputIndex.set(outputIndex, (this.textByOutputIndex.get(outputIndex) ?? "") + text);
					const block = this.output.content[outputIndex];
					if (block?.type === "text") block.text = this.textByOutputIndex.get(outputIndex) ?? "";
					events.push({ type: "text_delta", contentIndex: outputIndex, delta: text, partial: this.output });
				} else if (deltaType === "thinking_delta" && typeof outputIndex === "number") {
					const text = String(delta.thinking ?? "");
					this.thinkingByOutputIndex.set(
						outputIndex,
						(this.thinkingByOutputIndex.get(outputIndex) ?? "") + text,
					);
					const block = this.output.content[outputIndex];
					if (block?.type === "thinking") block.thinking = this.thinkingByOutputIndex.get(outputIndex) ?? "";
					events.push({
						type: "thinking_delta",
						contentIndex: outputIndex,
						delta: text,
						partial: this.output,
					});
				} else if (deltaType === "input_json_delta" && this.toolsByOutputIndex.size > 0) {
					const partialJson = String(delta.partial_json ?? "");
					const tool = this.toolsByOutputIndex.get(outputIndex);
					if (tool) {
						tool.argsJson += partialJson;
						this.toolsByOutputIndex.set(outputIndex, tool);
						events.push({
							type: "toolcall_delta",
							contentIndex: outputIndex,
							delta: partialJson,
							partial: this.output,
						});
					}
				}
				break;
			}
			case "content_block_stop": {
				const outputIndex = this.streamBlockIndexToOutputIndex.get(index);
				if (typeof outputIndex !== "number") break;

				if (this.textByOutputIndex.has(outputIndex)) {
					const block = this.output.content[outputIndex];
					const text = this.textByOutputIndex.get(outputIndex) ?? "";
					if (block?.type === "text") block.text = text;
					events.push({ type: "text_end", contentIndex: outputIndex, content: text, partial: this.output });
					this.textByOutputIndex.delete(outputIndex);
				}
				if (this.thinkingByOutputIndex.has(outputIndex)) {
					const block = this.output.content[outputIndex];
					const text = this.thinkingByOutputIndex.get(outputIndex) ?? "";
					if (block?.type === "thinking") block.thinking = text;
					events.push({
						type: "thinking_end",
						contentIndex: outputIndex,
						content: text,
						partial: this.output,
					});
					this.thinkingByOutputIndex.delete(outputIndex);
				}
				break;
			}
			case "message_stop": {
				for (const [, tool] of this.toolsByOutputIndex.entries()) {
					let parsedArgs: Record<string, unknown> = {};
					if (tool.argsJson.trim()) {
						try {
							parsedArgs = JSON.parse(tool.argsJson) as Record<string, unknown>;
						} catch {
							parsedArgs = {};
						}
					}
					const block = this.output.content[tool.contentIndex];
					if (block?.type === "toolCall") {
						block.arguments = parsedArgs;
					}
					events.push({
						type: "toolcall_end",
						contentIndex: tool.contentIndex,
						toolCall: {
							type: "toolCall",
							id: tool.id,
							name: tool.name,
							arguments: parsedArgs,
						},
						partial: this.output,
					});
				}
				this.toolsByOutputIndex.clear();
				break;
			}
		}

		return events;
	}

	private parseCompleteBlock(block: unknown): AssistantMessageEvent[] {
		const events: AssistantMessageEvent[] = [];
		if (!block || typeof block !== "object") return events;
		const typed = block as Record<string, unknown>;
		const blockType = String(typed.type ?? "");
		const contentIndex = this.output.content.length;

		if (blockType === "text") {
			const text = String(typed.text ?? "");
			this.output.content.push({ type: "text", text });
			events.push({ type: "text_start", contentIndex, partial: this.output });
			events.push({ type: "text_delta", contentIndex, delta: text, partial: this.output });
			events.push({ type: "text_end", contentIndex, content: text, partial: this.output });
		} else if (blockType === "thinking") {
			const text = String(typed.thinking ?? "");
			this.output.content.push({ type: "thinking", thinking: text });
			events.push({ type: "thinking_start", contentIndex, partial: this.output });
			events.push({ type: "thinking_delta", contentIndex, delta: text, partial: this.output });
			events.push({ type: "thinking_end", contentIndex, content: text, partial: this.output });
		} else if (blockType === "tool_use") {
			const id = String(typed.id ?? "");
			const name = String(typed.name ?? "tool");
			const args = (typed.input as Record<string, unknown>) ?? {};
			this.output.content.push({
				type: "toolCall",
				id,
				name,
				arguments: args as Record<string, unknown>,
			});
			events.push({ type: "toolcall_start", contentIndex, partial: this.output });
			events.push({
				type: "toolcall_end",
				contentIndex,
				toolCall: { type: "toolCall", id, name, arguments: args as Record<string, unknown> },
				partial: this.output,
			});
		}

		return events;
	}
}

class CodexCliParser {
	private output: AssistantMessage;
	private readonly model: Model<Api>;
	private finished = false;
	private errored = false;
	private nextContentIndex = 0;
	private readonly activeTools = new Map<
		string,
		{
			name: string;
			contentIndex: number;
			args: Record<string, unknown>;
		}
	>();

	constructor(model: Model<Api>) {
		this.output = createAssistantTemplate(model);
		this.model = model;
	}

	get isFinished(): boolean {
		return this.finished || this.errored;
	}

	get message(): AssistantMessage {
		return this.output;
	}

	get finalResult(): AssistantMessage {
		return this.output;
	}

	parseLine(msg: Record<string, unknown>): AssistantMessageEvent[] {
		if (this.isFinished) return [];
		const events: AssistantMessageEvent[] = [];
		const msgType = msg.type;
		const eventType = String(msgType ?? "");
		if (eventType === "thread.started" || eventType === "turn.started") {
			return events;
		}

		if (eventType === "item.started") {
			const item = (msg.item as Record<string, unknown>) || {};
			const itemType = String(item.type ?? "");
			const itemId = String(item.id ?? "");
			if (!itemId) return events;

			if (itemType === "command_execution") {
				const command = String(item.command ?? "");
				const contentIndex = this.nextContentIndex;
				this.nextContentIndex += 1;
				events.push({ type: "toolcall_start", contentIndex, partial: this.output });
				this.activeTools.set(itemId, {
					name: "shell",
					contentIndex,
					args: { command },
				});
				this.output.content.push({
					type: "toolCall",
					id: itemId,
					name: "shell",
					arguments: { command },
				});
			} else if (itemType === "file_change") {
				const pathValue = String(item.path ?? "");
				const contentIndex = this.nextContentIndex;
				this.nextContentIndex += 1;
				events.push({ type: "toolcall_start", contentIndex, partial: this.output });
				this.activeTools.set(itemId, {
					name: "file_edit",
					contentIndex,
					args: { path: pathValue },
				});
				this.output.content.push({
					type: "toolCall",
					id: itemId,
					name: "file_edit",
					arguments: { path: pathValue },
				});
			}
		} else if (eventType === "item.completed") {
			const item = (msg.item as Record<string, unknown>) || {};
			const itemType = String(item.type ?? "");
			const itemId = String(item.id ?? "");
			if (itemType === "agent_message") {
				const text = String(item.text ?? "");
				const contentIndex = this.nextContentIndex;
				this.nextContentIndex += 1;
				this.output.content.push({ type: "text", text: "" });
				events.push({ type: "text_start", contentIndex, partial: this.output });
				events.push({ type: "text_delta", contentIndex, delta: text, partial: this.output });
				this.output.content[this.nextContentIndex - 1] = { type: "text", text };
				events.push({ type: "text_end", contentIndex, content: text, partial: this.output });
			} else if (itemType === "command_execution" && itemId) {
				const details = this.activeTools.get(itemId);
				const name = details?.name ?? "shell";
				const contentIndex = details?.contentIndex ?? this.nextContentIndex;
				const args = details?.args ?? {};
				const command = String(item.command ?? args.command ?? "");
				if (details?.contentIndex === undefined) {
					this.nextContentIndex = Math.max(this.nextContentIndex, contentIndex + 1);
				}
				const commandArgs = command ? { ...args, command } : args;
				const block = this.output.content[contentIndex];
				if (block?.type === "toolCall") {
					block.arguments = commandArgs as Record<string, unknown>;
				}
				events.push({
					type: "toolcall_end",
					contentIndex,
					toolCall: {
						type: "toolCall",
						id: itemId,
						name,
						arguments: commandArgs as Record<string, unknown>,
					},
					partial: this.output,
				});
				this.activeTools.delete(itemId);
			} else if (itemType === "file_change" && itemId) {
				const details = this.activeTools.get(itemId);
				const name = details?.name ?? "file_edit";
				const contentIndex = details?.contentIndex ?? this.nextContentIndex;
				const pathValue = String(item.path ?? details?.args?.path ?? "");
				const args = { ...(details?.args ?? {}), path: pathValue };
				if (details?.contentIndex === undefined) {
					this.nextContentIndex = Math.max(this.nextContentIndex, contentIndex + 1);
				}
				const block = this.output.content[contentIndex];
				if (block?.type === "toolCall") {
					block.arguments = args;
				}
				events.push({
					type: "toolcall_end",
					contentIndex,
					toolCall: {
						type: "toolCall",
						id: itemId,
						name,
						arguments: args as Record<string, unknown>,
					},
					partial: this.output,
				});
				this.activeTools.delete(itemId);
			}
		} else if (eventType === "turn.completed") {
			applyUsage(this.output.usage, (msg.usage as Record<string, unknown>) ?? undefined);
			calculateCost(this.model, this.output.usage);
			this.finished = true;
			this.output.stopReason = "stop";
			events.push({ type: "done", reason: "stop", message: this.output });
		} else if (eventType === "turn.failed") {
			this.errored = true;
			this.output.stopReason = "error";
			this.output.errorMessage = String(msg.error ?? "Unknown error");
			events.push({ type: "error", reason: "error", error: this.output });
		}

		return events;
	}

	get outputMessage(): AssistantMessage {
		return this.output;
	}
}

function createAbortableProcess(
	command: string,
	args: string[],
	onLine: (line: string) => void,
	signal: AbortSignal | undefined,
): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		let settled = false;
		let proc: ReturnType<typeof spawn>;

		try {
			proc = spawn(command, args, {
				cwd: process.cwd(),
				env: { ...process.env },
				stdio: ["ignore", "pipe", "pipe"],
			});
		} catch (error) {
			reject(error as Error);
			return;
		}

		const settle = (error?: Error) => {
			if (settled) return;
			settled = true;
			if (error) reject(error);
			else resolve();
		};

		let buffer = "";

		const cleanup = () => {
			try {
				if (!proc.killed) proc.kill("SIGTERM");
			} catch {
				//
			}
		};

		proc.stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() || "";
			for (const raw of lines) {
				const line = raw.trim();
				if (!line) continue;
				onLine(line);
			}
		});

		proc.stderr.on("data", () => {
			// Intentionally ignore stderr to avoid blocking and keep stream focused on model output.
		});

		proc.on("error", (error) => {
			settle(error);
		});

		proc.on("close", (code) => {
			const remaining = buffer.trim();
			if (remaining) onLine(remaining);
			if (code !== 0 && !settled) {
				settle(new Error(`Process exited with code ${code}`));
				return;
			}
			settle();
		});

		signal?.addEventListener(
			"abort",
			() => {
				cleanup();
				settle(new DOMException("Request was aborted.", "AbortError"));
			},
			{ once: true },
		);

		if (signal?.aborted) {
			cleanup();
			settle(new DOMException("Request was aborted.", "AbortError"));
		}
	});
}

// ---------------------------------------------------------------------------
// Claude Code JSONL session helpers
// ---------------------------------------------------------------------------

function getClaudeSessionPath(sessionId: string, cwd: string): string {
	const resolvedCwd = resolve(cwd);
	const escapedCwd = resolvedCwd.replace(/\//g, "-");
	return join(homedir(), ".claude", "projects", escapedCwd, `${sessionId}.jsonl`);
}

function nowIso(): string {
	return new Date().toISOString().replace("+00:00", "Z");
}

function buildClaudeJsonl(context: Context, sessionId: string, cwd: string, model: string): string {
	const lines: string[] = [];
	const base = {
		isSidechain: false,
		userType: "external",
		cwd,
		sessionId,
		version: "2.0.55",
		gitBranch: "",
	};

	lines.push(JSON.stringify({ type: "summary", summary: "Session from pi claude-cli provider", leafUuid: randomUUID() }));

	let parentUuid: string | null = null;

	for (const msg of context.messages) {
		const msgUuid = randomUUID();
		const timestamp = nowIso();

		if (msg.role === "user") {
			const content = typeof msg.content === "string"
				? msg.content
				: (msg.content as Array<{ type: string; text?: string }>)
					.filter((b) => b.type === "text")
					.map((b) => b.text ?? "")
					.join("\n");
			lines.push(JSON.stringify({
				...base,
				type: "user",
				uuid: msgUuid,
				parentUuid,
				timestamp,
				message: { role: "user", content },
				thinkingMetadata: { level: "none", disabled: true, triggers: [] },
				todos: [],
			}));
		} else if (msg.role === "assistant") {
			const blocks = (msg.content as Array<{ type: string; text?: string; thinking?: string; id?: string; name?: string; arguments?: Record<string, unknown> }>)
				.flatMap((b) => {
					if (b.type === "text") return [{ type: "text", text: b.text ?? "" }];
					if (b.type === "toolCall") return [{
						type: "tool_use",
						id: b.id ?? randomUUID(),
						name: b.name ?? "unknown",
						input: b.arguments ?? {},
					}];
					// skip thinking — Claude requires a valid signature we can't reproduce
					return [];
				});
			lines.push(JSON.stringify({
				...base,
				type: "assistant",
				uuid: msgUuid,
				parentUuid,
				timestamp,
				message: {
					id: `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`,
					type: "message",
					role: "assistant",
					model,
					content: blocks,
					stop_reason: null,
					stop_sequence: null,
					usage: { input_tokens: 0, output_tokens: 0 },
				},
			}));
		} else if (msg.role === "toolResult") {
			const toolMsg = msg as { role: "toolResult"; toolCallId: string; toolName: string; content: Array<{ type: string; text?: string }>; isError: boolean };
			const content = toolMsg.content.map((c) => (c.type === "text" ? c.text ?? "" : "[image]")).join("\n");
			// Write dedicated tool_result entry (for pi display)
			const toolResultUuid = msgUuid;
			lines.push(JSON.stringify({
				...base,
				type: "tool_result",
				uuid: toolResultUuid,
				parentUuid,
				timestamp,
				tool_use_id: toolMsg.toolCallId,
				content,
				is_error: toolMsg.isError,
			}));
			// Write user-wrapped tool_result (for Claude API — standard Anthropic format)
			const wrappedUuid = randomUUID();
			lines.push(JSON.stringify({
				...base,
				type: "user",
				uuid: wrappedUuid,
				parentUuid: toolResultUuid,
				timestamp,
				message: {
					role: "user",
					content: [{
						tool_use_id: toolMsg.toolCallId,
						type: "tool_result",
						content,
						is_error: toolMsg.isError,
					}],
				},
				thinkingMetadata: { level: "none", disabled: true, triggers: [] },
				todos: [],
			}));
			// The parent for the next message should be the wrapped user message
			parentUuid = wrappedUuid;
			continue;
		}

		parentUuid = msgUuid;
	}

	return lines.join("\n") + "\n";
}

function writeClaudeSession(context: Context, sessionId: string, cwd: string, model: string): string {
	const path = getClaudeSessionPath(sessionId, cwd);
	mkdirSync(path.slice(0, path.lastIndexOf("/")), { recursive: true });
	writeFileSync(path, buildClaudeJsonl(context, sessionId, cwd, model), "utf8");
	return path;
}

// ---------------------------------------------------------------------------
// Session ID cache — persists across streamSimple calls for the same conversation
// Key: hash of the first user message content (stable across turns)
// Value: pi-generated session UUID (NOT Claude's session ID — we write our own JSONL)
// ---------------------------------------------------------------------------

const claudeSessionCache = new Map<string, string>();

function getConversationKey(context: Context): string {
	const firstUser = context.messages.find((m) => m.role === "user");
	if (!firstUser) return "no-user-message";
	const content = typeof firstUser.content === "string"
		? firstUser.content
		: (firstUser.content as Array<{ type: string; text?: string }>)
			.filter((b) => b.type === "text")
			.map((b) => b.text ?? "")
			.join("");
	// Simple djb2 hash — good enough for a cache key
	let hash = 5381;
	for (let i = 0; i < content.length; i++) {
		hash = ((hash << 5) + hash) ^ content.charCodeAt(i);
	}
	return (hash >>> 0).toString(16);
}

// ---------------------------------------------------------------------------
// Intercept-and-resume runner
// ---------------------------------------------------------------------------

/**
 * Spawns `claude --print --output-format stream-json` and returns when the
 * process emits a complete `assistant` message.  Kills the process immediately
 * after the first complete assistant message so Claude Code never executes
 * tools itself.  The caller is responsible for resuming via --resume.
 */
function spawnAndCollectAssistantMessage(
	args: string[],
	signal: AbortSignal | undefined,
): Promise<{ sessionId: string | null; assistantMsg: Record<string, unknown> | null; error: string | null }> {
	return new Promise((resolve) => {
		let proc: ReturnType<typeof spawn>;
		try {
			proc = spawn("claude", args, {
				cwd: process.cwd(),
				env: { ...process.env },
				stdio: ["ignore", "pipe", "pipe"],
			});
		} catch (err) {
			resolve({ sessionId: null, assistantMsg: null, error: String(err) });
			return;
		}

		let buffer = "";
		let sessionId: string | null = null;
		let assistantMsg: Record<string, unknown> | null = null;
		let settled = false;

		const finish = (error: string | null = null) => {
			if (settled) return;
			settled = true;
			try { if (!proc.killed) proc.kill("SIGTERM"); } catch { /* ignore */ }
			resolve({ sessionId, assistantMsg, error });
		};

		proc.stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const raw of lines) {
				const line = raw.trim();
				if (!line) continue;
				let obj: Record<string, unknown>;
				try { obj = JSON.parse(line) as Record<string, unknown>; } catch { continue; }

				if (obj.type === "system" && obj.subtype === "init") {
					sessionId = (obj.session_id as string) ?? null;
				}

				if (obj.type === "assistant") {
					const msg = obj.message as Record<string, unknown> | undefined;
					const content = Array.isArray(msg?.content) ? (msg!.content as unknown[]) : [];
					const hasToolUse = content.some((b) => (b as Record<string, unknown>).type === "tool_use");
					if (hasToolUse) {
						assistantMsg = obj;
						finish();
						return;
					}
				}

				if (obj.type === "result") {
					// No tool calls — Claude finished cleanly.
					// We don't need to kill; process is already done.
					finish();
					return;
				}
			}
		});

		proc.stderr.on("data", () => { /* ignore */ });
		proc.on("error", (err) => finish(err.message));
		proc.on("close", () => finish());

		signal?.addEventListener("abort", () => finish("aborted"), { once: true });
		if (signal?.aborted) finish("aborted");
	});
}

async function runClaudeCliStream(
	model: Model<Api>,
	cliModel: string,
	context: Context,
	outputStream: AssistantMessageEventStream,
	signal?: AbortSignal,
): Promise<void> {
	const initialParser = new ClaudeCliParser(model);
	outputStream.push({ type: "start", partial: initialParser.message });

	const cwd = process.cwd();
	const conversationKey = getConversationKey(context);
	let resumeSessionId: string | null = claudeSessionCache.get(conversationKey) ?? null;

	// Build base CLI args (no prompt yet — added per-call below)
	const baseArgs = [
		"--print",
		"--verbose",
		"--output-format", "stream-json",
		"--include-partial-messages",
		"--model", cliModel,
	];

	// Check for tool calls in context — if present, this is a resume-after-tools call.
	const hasPriorToolResults = context.messages.some((m) => m.role === "toolResult");

	// Build args for this call
	if (resumeSessionId !== null) {
		// Resume: write full context (including new tool results) to JSONL, then
		// run with --input-format stream-json and send [continue] via stdin.
		writeClaudeSession(context, resumeSessionId, cwd, cliModel);
		const resumeArgs = [
			...baseArgs,
			"--input-format", "stream-json",
			"--resume", resumeSessionId,
		];

		const resumeParser = new ClaudeCliParser(model);
		const continueMsg = JSON.stringify({ type: "user", message: { role: "user", content: "[continue]" } }) + "\n";
		try {
			await new Promise<void>((resolve, reject) => {
				let proc: ReturnType<typeof spawn>;
				try {
					proc = spawn("claude", resumeArgs, {
						cwd,
						env: { ...process.env },
						stdio: ["pipe", "pipe", "pipe"],
					});
				} catch (err) {
					reject(err as Error);
					return;
				}
				let settled = false;
				const settle = (err?: Error) => {
					if (settled) return;
					settled = true;
					if (err) reject(err); else resolve();
				};

				let buf = "";
				proc.stdout.on("data", (chunk: Buffer) => {
					buf += chunk.toString();
					const lines = buf.split("\n");
					buf = lines.pop() ?? "";
					for (const raw of lines) {
						const line = raw.trim();
						if (!line) continue;
						try {
							const parsed = JSON.parse(line) as Record<string, unknown>;
							for (const event of resumeParser.parseLine(parsed)) outputStream.push(event);
						} catch { /* ignore */ }
					}
				});
				proc.stderr.on("data", () => { /* ignore */ });
				proc.on("error", (err) => settle(err));
				proc.on("close", () => {
					const remaining = buf.trim();
					if (remaining) {
						try {
							const parsed = JSON.parse(remaining) as Record<string, unknown>;
							for (const event of resumeParser.parseLine(parsed)) outputStream.push(event);
						} catch { /* ignore */ }
					}
					settle();
				});
				signal?.addEventListener("abort", () => {
					try { if (!proc.killed) proc.kill("SIGTERM"); } catch { /* ignore */ }
					settle(new DOMException("aborted", "AbortError"));
				}, { once: true });
				if (signal?.aborted) {
					try { if (!proc.killed) proc.kill("SIGTERM"); } catch { /* ignore */ }
					settle(new DOMException("aborted", "AbortError"));
					return;
				}
				// Send [continue] trigger via stdin, then close stdin to signal EOF
				proc.stdin.write(continueMsg, () => {
					proc.stdin.end();
				});
			});
			if (!resumeParser.isFinished) {
				outputStream.push({ type: "error", reason: "error", error: resumeParser.finalResult });
			}
		} catch (err) {
			if (err instanceof DOMException && err.name === "AbortError") {
				resumeParser.finalResult.stopReason = "aborted";
				outputStream.push({ type: "error", reason: "aborted", error: resumeParser.finalResult });
			} else {
				resumeParser.finalResult.stopReason = "error";
				resumeParser.finalResult.errorMessage = err instanceof Error ? err.message : String(err);
				outputStream.push({ type: "error", reason: "error", error: resumeParser.finalResult });
			}
		}
		outputStream.end();
		return;
	}

	if (!hasPriorToolResults) {
		// First call: use flattened prompt, intercept if tool calls appear
		const { sessionId, assistantMsg, error } = await spawnAndCollectAssistantMessage(
			[...baseArgs, ...(context.systemPrompt?.trim() ? ["--system-prompt", context.systemPrompt.trim()] : []), formatContextPrompt(context)],
			signal,
		);

		if (sessionId) {
			// Use a pi-generated UUID for our session file (not Claude's session ID).
			// Claude's session file only has queue-operations — we write our own JSONL.
			resumeSessionId = randomUUID();
			claudeSessionCache.set(conversationKey, resumeSessionId);
		}

		if (error === "aborted") {
			initialParser.finalResult.stopReason = "aborted";
			initialParser.finalResult.errorMessage = "Request was aborted.";
			outputStream.push({ type: "error", reason: "aborted", error: initialParser.finalResult });
			outputStream.end();
			return;
		}

		if (error) {
			initialParser.finalResult.stopReason = "error";
			initialParser.finalResult.errorMessage = error;
			outputStream.push({ type: "error", reason: "error", error: initialParser.finalResult });
			outputStream.end();
			return;
		}

		if (assistantMsg !== null) {
			// Got tool calls — parse and emit, then stop so pi can run them
			for (const event of initialParser.parseLine(assistantMsg)) {
				outputStream.push(event);
			}
			const toolCalls = (initialParser.finalResult.content as Array<{ type: string }>).filter((c) => c.type === "toolCall");
			if (toolCalls.length > 0) {
				outputStream.push({ type: "done", reason: "toolUse", message: initialParser.finalResult });
				outputStream.end();
				return;
			}
		}

		// No tool calls on first turn — run a fresh full stream (don't resume)
		const freshParser = new ClaudeCliParser(model);
		const freshArgs = [...baseArgs];
		if (context.systemPrompt?.trim()) freshArgs.push("--system-prompt", context.systemPrompt.trim());
		freshArgs.push(formatContextPrompt(context));
		try {
			await createAbortableProcess("claude", freshArgs, (line) => {
				try {
					const parsed = JSON.parse(line) as Record<string, unknown>;
					for (const event of freshParser.parseLine(parsed)) outputStream.push(event);
				} catch { /* ignore */ }
			}, signal);
			if (!freshParser.isFinished) outputStream.push({ type: "error", reason: "error", error: freshParser.finalResult });
		} catch (err) {
			if (err instanceof DOMException && err.name === "AbortError") {
				freshParser.finalResult.stopReason = "aborted";
				outputStream.push({ type: "error", reason: "aborted", error: freshParser.finalResult });
			} else {
				freshParser.finalResult.stopReason = "error";
				freshParser.finalResult.errorMessage = err instanceof Error ? err.message : String(err);
				outputStream.push({ type: "error", reason: "error", error: freshParser.finalResult });
			}
		}
		outputStream.end();
		return;
	}

	outputStream.end();
}

async function runCodexCliStream(
	command: string[],
	parser: CodexCliParser,
	outputStream: AssistantMessageEventStream,
	signal?: AbortSignal,
): Promise<void> {
	const binary = command[0];
	const args = command.slice(1);

	outputStream.push({ type: "start", partial: parser.message });

	try {
		await createAbortableProcess(
			binary,
			args,
			(line) => {
				try {
					const parsed = JSON.parse(line) as Record<string, unknown>;
					for (const event of parser.parseLine(parsed)) {
						outputStream.push(event);
					}
				} catch {
					// Ignore malformed JSON lines.
				}
			},
			signal,
		);

		if (!parser.isFinished) {
			for (const event of parser.parseLine({
				type: "turn.failed",
				error: "Codex CLI stream ended unexpectedly.",
			})) {
				outputStream.push(event);
			}
		}
	} catch (error) {
		if (error instanceof DOMException && error.name === "AbortError") {
			parser.outputMessage.stopReason = "aborted";
			parser.outputMessage.errorMessage = "Request was aborted.";
			outputStream.push({ type: "error", reason: "aborted", error: parser.outputMessage });
		} else if (!parser.isFinished) {
			parser.outputMessage.stopReason = "error";
			parser.outputMessage.errorMessage = error instanceof Error ? error.message : String(error);
			outputStream.push({ type: "error", reason: "error", error: parser.outputMessage });
		}
	} finally {
		outputStream.end();
	}
}

function streamClaudeCli(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const parsedModel = CLAUDE_MODELS_BY_ID.get(model.id);
	const cliModel = parsedModel?.apiModel ?? model.id;
	void runClaudeCliStream(model, cliModel, context, stream, options?.signal);
	return stream;
}

function streamCodexCli(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const parsedModel = CODEX_MODELS_BY_ID.get(model.id);
	const cliModel = parsedModel?.apiModel ?? model.id;
	const parser = new CodexCliParser(model);

	const prompt = formatContextPrompt(context);
	const cmd = [
		"codex",
		"exec",
		"--json",
		"--skip-git-repo-check",
		"--sandbox",
		"workspace-write",
		"--model",
		cliModel,
		prompt,
	];

	void runCodexCliStream(cmd, parser, stream, options?.signal);
	return stream;
}

export default function (pi: ExtensionAPI) {
	// Register Claude Code's TitleCase tool names so pi can execute them when
	// the intercept-and-resume loop hands control back from Claude Code.
	// Parameter names differ between Claude Code and pi's built-in tools,
	// so each wrapper adapts the schema at the boundary.

	pi.registerTool({
		name: "Bash",
		label: "Bash",
		description: "Execute a bash command",
		parameters: Type.Object({
			command: Type.String({ description: "Bash command to execute" }),
			description: Type.Optional(Type.String({ description: "Description of command" })),
			timeout: Type.Optional(Type.Number({ description: "Timeout in milliseconds" })),
		}),
		execute: (id, params, signal, onUpdate) =>
			bashTool.execute(id, { command: params.command, timeout: params.timeout }, signal, onUpdate),
	});

	pi.registerTool({
		name: "Read",
		label: "Read",
		description: "Read a file",
		parameters: Type.Object({
			file_path: Type.String({ description: "Absolute path to the file" }),
			offset: Type.Optional(Type.Number({ description: "Line number to start reading from" })),
			limit: Type.Optional(Type.Number({ description: "Number of lines to read" })),
		}),
		execute: (id, params, signal, onUpdate) =>
			readTool.execute(id, { path: params.file_path, offset: params.offset, limit: params.limit }, signal, onUpdate),
	});

	pi.registerTool({
		name: "Write",
		label: "Write",
		description: "Write a file",
		parameters: Type.Object({
			file_path: Type.String({ description: "Absolute path to the file" }),
			content: Type.String({ description: "Content to write" }),
		}),
		execute: (id, params, signal, onUpdate) =>
			writeTool.execute(id, { path: params.file_path, content: params.content }, signal, onUpdate),
	});

	pi.registerTool({
		name: "Edit",
		label: "Edit",
		description: "Edit a file by replacing text",
		parameters: Type.Object({
			file_path: Type.String({ description: "Absolute path to the file" }),
			old_string: Type.String({ description: "Exact text to find and replace" }),
			new_string: Type.String({ description: "New text to replace with" }),
		}),
		execute: (id, params, signal, onUpdate) =>
			editTool.execute(id, { path: params.file_path, oldText: params.old_string, newText: params.new_string }, signal, onUpdate),
	});

	pi.registerTool({
		name: "Glob",
		label: "Glob",
		description: "Find files matching a glob pattern",
		parameters: Type.Object({
			pattern: Type.String({ description: "Glob pattern to match files" }),
			path: Type.Optional(Type.String({ description: "Directory to search in" })),
		}),
		execute: (id, params, signal, onUpdate) =>
			findTool.execute(id, { pattern: params.pattern, path: params.path }, signal, onUpdate),
	});

	pi.registerTool({
		name: "Grep",
		label: "Grep",
		description: "Search file contents for a pattern",
		parameters: Type.Object({
			pattern: Type.String({ description: "Search pattern (regex)" }),
			path: Type.Optional(Type.String({ description: "Directory or file to search" })),
			glob: Type.Optional(Type.String({ description: "Filter files by glob pattern" })),
			"-i": Type.Optional(Type.Boolean({ description: "Case insensitive search" })),
		}),
		execute: (id, params, signal, onUpdate) =>
			grepTool.execute(id, { pattern: params.pattern, path: params.path, glob: params.glob, ignoreCase: params["-i"] }, signal, onUpdate),
	});

	pi.registerProvider("claude-cli", {
		baseUrl: "https://localhost/claude-cli",
		apiKey: PROVIDER_API_KEY_PLACEHOLDER,
		api: CLAUDE_CLI_PROVIDER_API,
		models: CLAUDE_CLI_MODELS.map(({ id, name, reasoning, input, cost, contextWindow, maxTokens }) => ({
			id,
			name,
			reasoning,
			input,
			cost,
			contextWindow,
			maxTokens,
		})),
		streamSimple: streamClaudeCli,
	});

	pi.registerProvider("codex-cli", {
		baseUrl: "https://localhost/codex-cli",
		apiKey: PROVIDER_API_KEY_PLACEHOLDER,
		api: CODEX_CLI_PROVIDER_API,
		models: CODEX_CLI_MODELS.map(({ id, name, reasoning, input, cost, contextWindow, maxTokens }) => ({
			id,
			name,
			reasoning,
			input,
			cost,
			contextWindow,
			maxTokens,
		})),
		streamSimple: streamCodexCli,
	});
}
