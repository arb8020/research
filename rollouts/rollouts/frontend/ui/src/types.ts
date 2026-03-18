// Rollouts trace viewer types
// Mirrors wafer-app TraceSample/TraceReport shapes where compatible

export interface RunReport {
  eval_name: string;
  dataset_path: string;
  total_samples: number;
  summary_metrics: {
    mean_reward: number;
    min_reward: number;
    max_reward: number;
    std_reward?: number;
    total_samples: number;
    avg_turns?: number;
    avg_tokens?: number;
    provider_errors?: number;
    failed_samples?: number;
    successful_samples?: number;
    success_rate?: number;
    completion_rate?: number;
    [key: string]: number | undefined;
  };
  config: {
    endpoint: {
      provider: string;
      model: string;
      api_base?: string;
      max_tokens?: number;
      temperature?: number;
    };
    max_samples?: number;
    max_concurrent?: number;
    evaluation_timestamp?: string;
  };
  timestamp: string;
  git_info?: {
    commit: string;
    branch: string;
    dirty: boolean;
  };
  sample_ids: string[];
}

export interface RunListItem {
  id: string;
  name: string;
  timestamp: number;
  total_samples: number;
  mean_reward: number;
}

export interface SampleReward {
  name: string;
  value: number;
  weight: number;
  metadata: Record<string, unknown>;
}

export interface SampleMetadata {
  turns_used?: number;
  stop_reason?: string | null;
  total_tokens?: number;
  status?: string;
  duration_seconds?: number;
  [key: string]: unknown;
}

export interface SampleCompletion {
  id: string;
  object: string;
  created: number;
  model: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
    reasoning_tokens?: number;
    cache_read_tokens?: number;
    cache_write_tokens?: number;
    cost?: {
      input: number;
      output: number;
      cache_read?: number;
      cache_write?: number;
    };
  };
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: unknown;
      tool_call_id?: string | null;
    };
    finish_reason: string;
    stop_reason?: string | null;
  }>;
}

export interface TraceSample {
  id: string;
  index: number | null;
  group_index: number | null;
  input: Record<string, unknown> | null;  // null in charisma (uses 'problem' instead)
  prompt: string;
  ground_truth: string | null;
  reward?: number;                         // scalar reward (charisma)
  trajectory: {
    completions: SampleCompletion[];
    messages?: Array<{
      role: string;
      content: string | unknown[];
      tool_call_id?: string | null;
      [key: string]: unknown;
    }>;
  };
  rewards?: SampleReward[];
  score?: { metrics: SampleReward[] };    // charisma score envelope
  environment_state?: unknown;
  status?: string;
  metadata?: SampleMetadata;
}

// Workspace snapshot types (from /workspace endpoint)

export interface BashHistoryEntry {
  turn: number
  cmd: string
  stdout: string
  stderr: string
  exit_code: number
  uncertain_fs_effects: boolean
}

export interface WorkspaceSnapshot {
  turn: number
  files: Record<string, string>   // filename -> contents
  cwd: string
  bash_history: BashHistoryEntry[]
}

export interface LineEdit {
  turn: number
  type: 'edit' | 'write' | 'bash_sed' | 'bash_redirect' | 'patch'
  diff: string
  message_index: number
  cmd: string
}

// line_history[filename][line_number_str] = LineEdit[]
export type LineHistory = Record<string, Record<string, LineEdit[]>>

export interface WorkspaceData {
  snapshots: WorkspaceSnapshot[]
  line_history: LineHistory
  source: 'live' | 'reconstructed'
}

// Live run types

export interface LiveRun {
  run_id: string;
  config_name: string;
  start_time: number;
  status: 'running' | 'completed' | 'failed' | 'killed';
  exit_code: number | null;
  output_length?: number;
}

export type StreamEvent =
  | { type: 'eval_start'; name: string; total: number; timestamp: string }
  | { type: 'sample_start'; id: string; name: string; timestamp: string }
  | { type: 'turn'; id: string; turn: number; status: string; timestamp: string }
  | { type: 'sample_end'; id: string; score: number; timestamp: string }
  | { type: 'eval_end'; name: string; total: number; timestamp: string }
  | { type: 'stdout'; line: string; timestamp?: string }
  | { type: 'complete'; exit_code: number; status: string }

// Derived types for live run tracking in UI

export interface LiveSample {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'done';
  turn: number;
  score: number | null;
}

export interface LiveRunState {
  run_id: string;
  config_name: string;
  start_time: number;
  status: 'running' | 'completed' | 'failed' | 'killed';
  total: number | null;
  samples: Map<string, LiveSample>;
  stdout_lines: string[];
}
