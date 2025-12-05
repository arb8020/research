#!/usr/bin/env python3
"""Pandas-based NCU CSV analysis utilities.

Programmatic analysis of NVIDIA Nsight Compute CSV reports to identify
performance bottlenecks and optimization opportunities.

Inspired by qwen3_next/src/profiling/trace_analysis.py

Usage:
    # Analyze single report
    python analyze_ncu.py ncu_reports_remote/cute_tiny_single_ncu.csv

    # Compare two backends
    python analyze_ncu.py ncu_reports_remote/reference_tiny_single_ncu.csv \\
        --compare ncu_reports_remote/cute_tiny_single_ncu.csv

    # Save summary to CSV
    python analyze_ncu.py ncu_reports_remote/cute_tiny_single_ncu.csv --output summary.csv

    # As a library
    import analyze_ncu as ncu
    df, err = ncu.load_ncu_csv("report.csv")
    summary = ncu.summarize_kernels(ncu.analyze_kernel_performance(df))
"""

import sys
from pathlib import Path

import pandas as pd


def load_ncu_csv(path: Path | str) -> tuple[pd.DataFrame, None] | tuple[None, str]:
    """Load NCU CSV report into Pandas DataFrame.

    Tiger Style: Assert preconditions, return tuple for errors.

    Args:
        path: Path to NCU CSV file

    Returns:
        (DataFrame, None) on success
        (None, error_message) on failure
    """
    # Assert preconditions
    assert isinstance(path, (Path, str)), "path must be Path or str"
    path = Path(path)

    if not path.exists():
        return None, f"NCU CSV file not found: {path}"

    if not path.is_file():
        return None, f"Path is not a file: {path}"

    try:
        # NCU CSV has ==PROF== header lines we need to skip
        # Find the first line that starts with "ID" (header line)
        with open(path) as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"'):
                header_idx = i
                break

        if header_idx is None:
            return None, "Could not find CSV header in NCU file"

        # Read CSV starting from header
        from io import StringIO
        csv_data = ''.join(lines[header_idx:])
        df = pd.read_csv(StringIO(csv_data))

    except Exception as e:
        return None, f"Failed to load NCU CSV: {e}"

    if df.empty:
        return None, "NCU CSV contains no data"

    # Verify expected columns
    required_cols = ['Kernel Name', 'Metric Name', 'Metric Value']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"

    return df, None


def pivot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot NCU data so each row is a kernel with metrics as columns.

    Args:
        df: DataFrame from load_ncu_csv()

    Returns:
        DataFrame with kernels as rows, metrics as columns
    """
    # Each kernel invocation (ID) has multiple metric rows
    # Pivot so each kernel is one row with metrics as columns
    pivoted = df.pivot_table(
        index=['ID', 'Kernel Name', 'Block Size', 'Grid Size'],
        columns='Metric Name',
        values='Metric Value',
        aggfunc='first'
    ).reset_index()

    return pivoted


def analyze_kernel_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze kernel performance from NCU metrics.

    Args:
        df: DataFrame from load_ncu_csv()

    Returns:
        DataFrame with performance analysis per kernel
    """
    # Pivot metrics
    kernel_df = pivot_metrics(df)

    # Common metric columns (rename for convenience)
    metric_map = {
        'gpu__time_duration.sum': 'duration_ns',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm_utilization_pct',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'dram_utilization_pct',
    }

    # Rename columns if they exist
    for old_name, new_name in metric_map.items():
        if old_name in kernel_df.columns:
            kernel_df = kernel_df.rename(columns={old_name: new_name})

    # Calculate derived metrics
    if 'duration_ns' in kernel_df.columns:
        kernel_df['duration_us'] = kernel_df['duration_ns'] / 1000.0
        kernel_df['duration_ms'] = kernel_df['duration_ns'] / 1_000_000.0

    return kernel_df


def summarize_kernels(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance across all kernel invocations.

    Groups by kernel name and aggregates metrics.

    Args:
        df: DataFrame from analyze_kernel_performance()

    Returns:
        Summary DataFrame with aggregated stats per kernel
    """
    # Group by kernel name
    agg_funcs = {}

    if 'duration_ns' in df.columns:
        agg_funcs['duration_ns'] = ['count', 'mean', 'sum', 'min', 'max']
    if 'sm_utilization_pct' in df.columns:
        agg_funcs['sm_utilization_pct'] = ['mean', 'min', 'max']
    if 'dram_utilization_pct' in df.columns:
        agg_funcs['dram_utilization_pct'] = ['mean', 'min', 'max']

    if not agg_funcs:
        return pd.DataFrame()

    summary = df.groupby('Kernel Name').agg(agg_funcs)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Sort by total time
    if 'duration_ns_sum' in summary.columns:
        summary = summary.sort_values('duration_ns_sum', ascending=False)

        # Calculate percentage of total time
        total_time = summary['duration_ns_sum'].sum()
        summary['pct_total_time'] = (summary['duration_ns_sum'] / total_time * 100).round(2)

    return summary.reset_index()


def find_bottlenecks(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find kernels with performance bottlenecks.

    Identifies kernels with low SM or DRAM utilization.

    Args:
        df: DataFrame from analyze_kernel_performance()
        top_n: Number of bottlenecks to return

    Returns:
        DataFrame with bottleneck kernels
    """
    bottlenecks = []

    # Low SM utilization
    if 'sm_utilization_pct' in df.columns and 'duration_us' in df.columns:
        low_sm = df[df['sm_utilization_pct'] < 50.0].copy()
        if not low_sm.empty:
            low_sm['bottleneck_type'] = 'Low SM Utilization'
            low_sm['severity'] = low_sm['duration_us'] * (50 - low_sm['sm_utilization_pct'])
            bottlenecks.append(low_sm)

    # Low DRAM utilization (but high duration - indicates compute bound)
    if 'dram_utilization_pct' in df.columns and 'duration_us' in df.columns:
        low_dram = df[(df['dram_utilization_pct'] < 10.0) & (df['duration_us'] > 10)].copy()
        if not low_dram.empty:
            low_dram['bottleneck_type'] = 'Low DRAM Utilization'
            low_dram['severity'] = low_dram['duration_us']
            bottlenecks.append(low_dram)

    if not bottlenecks:
        return pd.DataFrame()

    all_bottlenecks = pd.concat(bottlenecks, ignore_index=True)
    all_bottlenecks = all_bottlenecks.sort_values('severity', ascending=False)

    return all_bottlenecks.head(top_n)


def compare_backends(
    baseline_path: Path | str,
    compare_path: Path | str
) -> tuple[dict, None] | tuple[None, str]:
    """Compare NCU metrics between two backend implementations.

    Args:
        baseline_path: Path to baseline NCU CSV
        compare_path: Path to comparison NCU CSV

    Returns:
        (comparison_dict, None) on success
        (None, error_message) on failure
    """
    baseline_path = Path(baseline_path)
    compare_path = Path(compare_path)

    # Load both CSVs
    baseline_df, err = load_ncu_csv(baseline_path)
    if err:
        return None, f"Baseline: {err}"

    compare_df, err = load_ncu_csv(compare_path)
    if err:
        return None, f"Compare: {err}"

    # Analyze both
    baseline_perf = analyze_kernel_performance(baseline_df)
    compare_perf = analyze_kernel_performance(compare_df)

    # Calculate totals
    baseline_total_ns = baseline_perf['duration_ns'].sum() if 'duration_ns' in baseline_perf.columns else 0
    compare_total_ns = compare_perf['duration_ns'].sum() if 'duration_ns' in compare_perf.columns else 0

    speedup = baseline_total_ns / compare_total_ns if compare_total_ns > 0 else 0

    # Build comparison
    comparison = {
        'baseline_file': str(baseline_path),
        'compare_file': str(compare_path),
        'baseline_total_us': baseline_total_ns / 1000.0,
        'compare_total_us': compare_total_ns / 1000.0,
        'speedup': speedup,
        'baseline_kernels': len(baseline_perf),
        'compare_kernels': len(compare_perf),
    }

    # Average utilization
    if 'sm_utilization_pct' in baseline_perf.columns:
        comparison['baseline_avg_sm_util'] = baseline_perf['sm_utilization_pct'].mean()
    if 'sm_utilization_pct' in compare_perf.columns:
        comparison['compare_avg_sm_util'] = compare_perf['sm_utilization_pct'].mean()

    if 'dram_utilization_pct' in baseline_perf.columns:
        comparison['baseline_avg_dram_util'] = baseline_perf['dram_utilization_pct'].mean()
    if 'dram_utilization_pct' in compare_perf.columns:
        comparison['compare_avg_dram_util'] = compare_perf['dram_utilization_pct'].mean()

    return comparison, None


def main():
    """CLI for NCU CSV analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze NCU CSV reports")
    parser.add_argument('ncu_csv', type=Path, help='Path to NCU CSV file')
    parser.add_argument('--compare', type=Path, help='Compare against another NCU CSV')
    parser.add_argument('--output', type=Path, help='Save summary to CSV')

    args = parser.parse_args()

    # Load and analyze
    df, err = load_ncu_csv(args.ncu_csv)
    if err:
        print(f"❌ Error: {err}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 80}")
    print(f"NCU ANALYSIS: {args.ncu_csv.name}")
    print(f"{'=' * 80}\n")

    # Analyze performance
    perf_df = analyze_kernel_performance(df)
    print(f"Total kernel invocations: {len(perf_df)}")

    if 'duration_ns' in perf_df.columns:
        total_us = perf_df['duration_ns'].sum() / 1000.0
        print(f"Total GPU time: {total_us:.3f} μs ({total_us / 1000:.3f} ms)")

    # Summarize by kernel
    summary = summarize_kernels(perf_df)
    if not summary.empty:
        print(f"\n{'=' * 80}")
        print("KERNEL SUMMARY (by total time)")
        print(f"{'=' * 80}\n")

        # Prepare display columns
        display_summary = summary.head(15).copy()

        # Truncate kernel names for display
        display_summary['Kernel Name'] = display_summary['Kernel Name'].apply(
            lambda x: x[:60] + '...' if len(x) > 60 else x
        )

        # Select and rename columns for display
        display_cols = {
            'Kernel Name': 'Kernel Name',
        }
        if 'duration_ns_count' in summary.columns:
            display_cols['duration_ns_count'] = 'Calls'
        if 'duration_ns_mean' in summary.columns:
            display_cols['duration_ns_mean'] = 'Avg (ns)'
        if 'pct_total_time' in summary.columns:
            display_cols['pct_total_time'] = '% Time'
        if 'sm_utilization_pct_mean' in summary.columns:
            display_cols['sm_utilization_pct_mean'] = 'SM %'
        if 'dram_utilization_pct_mean' in summary.columns:
            display_cols['dram_utilization_pct_mean'] = 'DRAM %'

        display_summary = display_summary[list(display_cols.keys())].rename(columns=display_cols)
        print(display_summary.to_string(index=False))

    # Find bottlenecks
    bottlenecks = find_bottlenecks(perf_df, top_n=10)
    if not bottlenecks.empty:
        print(f"\n{'=' * 80}")
        print("PERFORMANCE BOTTLENECKS")
        print(f"{'=' * 80}\n")

        for _, row in bottlenecks.iterrows():
            kernel_name = row['Kernel Name'][:70]
            btype = row['bottleneck_type']
            duration = row.get('duration_us', 0)
            sm_util = row.get('sm_utilization_pct', 0)
            dram_util = row.get('dram_utilization_pct', 0)

            print(f"• {kernel_name}")
            print(f"  Type: {btype}, Duration: {duration:.2f} μs")
            print(f"  SM: {sm_util:.1f}%, DRAM: {dram_util:.1f}%")
            print()

    # Compare if requested
    if args.compare:
        if args.compare.exists():
            print(f"\n{'=' * 80}")
            print("COMPARISON")
            print(f"{'=' * 80}\n")

            comparison, err = compare_backends(args.ncu_csv, args.compare)
            if err:
                print(f"❌ Comparison error: {err}")
            else:
                print(f"Baseline: {comparison['baseline_file']}")
                print(f"Compare:  {comparison['compare_file']}\n")

                print(f"Baseline total time: {comparison['baseline_total_us']:.3f} μs")
                print(f"Compare total time:  {comparison['compare_total_us']:.3f} μs")

                speedup = comparison['speedup']
                if speedup > 1:
                    print(f"Speedup: {speedup:.3f}x ({(speedup - 1) * 100:.1f}% faster)")
                elif speedup < 1 and speedup > 0:
                    print(f"Slowdown: {1 / speedup:.3f}x ({(1 - speedup) * 100:.1f}% slower)")

                if 'baseline_avg_sm_util' in comparison:
                    print(f"\nBaseline avg SM utilization: {comparison['baseline_avg_sm_util']:.1f}%")
                    print(f"Compare avg SM utilization:  {comparison['compare_avg_sm_util']:.1f}%")

                if 'baseline_avg_dram_util' in comparison:
                    print(f"\nBaseline avg DRAM utilization: {comparison['baseline_avg_dram_util']:.1f}%")
                    print(f"Compare avg DRAM utilization:  {comparison['compare_avg_dram_util']:.1f}%")
        else:
            print(f"\n⚠️  Compare file not found: {args.compare}")

    # Save if requested
    if args.output and not summary.empty:
        summary.to_csv(args.output, index=False)
        print(f"\n✅ Summary saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
