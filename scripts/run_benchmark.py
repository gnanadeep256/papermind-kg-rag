import argparse
import sys
from src.evaluation.benchmark_runner import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Evaluation & Benchmarking Runner")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live benchmark calling Gemini/Groq APIs and Vector/Graph Retrievers (default is dry-run mode)"
    )
    args = parser.parse_args()
    
    # Run in dry-run mode unless --live is explicitly requested
    dry_run = not args.live
    mode_str = "DRY-RUN (Mock LLM/Retrievers)" if dry_run else "LIVE (Gemini/Groq APIs)"
    print(f"Initializing GraphRAG Benchmark in {mode_str} mode...")
    
    try:
        runner = BenchmarkRunner(dry_run=dry_run)
        report_path = runner.run_benchmark()
        print(f"\n=======================================================")
        print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"Report generated: {report_path}")
        print(f"=======================================================")
    except Exception as e:
        print(f"Benchmark run failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
