"""
Fireworks-specific benchmark service for performance testing
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from src.modules.llm_completion import FireworksBenchmark, FireworksConfig
from src.logger import logger


@dataclass
class BenchmarkRequest:
    """Request configuration for benchmark"""

    model_key: str
    prompt: str
    concurrency: int = 10
    max_tokens: int = 256
    temperature: float = 0.7
    test_duration: Optional[int] = None  # seconds


@dataclass
class BenchmarkResult:
    """Detailed benchmark results"""

    model_name: str
    model_id: str
    concurrency: int
    prompt: str

    # Timing metrics
    total_time: float
    avg_time_to_first_token: float

    # Throughput metrics
    avg_tokens_per_second: float
    aggregate_tokens_per_second: float
    peak_tokens_per_second: float

    # Success metrics
    total_requests: int
    successful_requests: int
    error_rate: float

    # Token metrics
    total_tokens_generated: int
    avg_tokens_per_request: float

    # Quality metrics
    sample_completion: str
    completion_lengths: List[int]

    # Raw data
    individual_results: List[Dict[str, Any]]
    error_messages: List[str]

    # Metadata
    timestamp: float
    config_used: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        return (1.0 - self.error_rate) * 100

    @property
    def avg_completion_length(self) -> float:
        """Average completion length in characters"""
        if self.completion_lengths:
            return sum(self.completion_lengths) / len(self.completion_lengths)
        return 0


class FireworksBenchmarkService:
    """Service for running comprehensive Fireworks benchmarks"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.config = FireworksConfig()
        self.benchmark = FireworksBenchmark(api_key)

    async def run_single_benchmark(
        self,
        request: BenchmarkRequest,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark test

        Args:
            request: Benchmark configuration
            progress_callback: Optional callback for progress updates

        Returns:
            BenchmarkResult with comprehensive metrics
        """
        model_config = self.config.get_model(request.model_key)
        start_time = time.time()

        if progress_callback:
            progress_callback(
                "starting",
                {
                    "model": model_config["name"],
                    "concurrency": request.concurrency,
                    "status": "Initializing benchmark...",
                },
            )

        try:
            # Run the concurrent benchmark
            raw_results = await self.benchmark.run_concurrent_benchmark(
                model_key=request.model_key,
                prompt=request.prompt,
                concurrency=request.concurrency,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            # Extract individual results and calculate advanced metrics
            individual_results = raw_results.get("individual_results", [])
            successful_results = [
                r for r in individual_results if r.get("tokens", 0) > 0
            ]
            error_results = [r for r in individual_results if "error" in r]

            # Calculate peak TPS
            peak_tps = (
                max([r.get("tps", 0) for r in successful_results])
                if successful_results
                else 0
            )

            # Calculate completion lengths
            completion_lengths = [
                len(r.get("completion_text", "")) for r in successful_results
            ]

            # Extract error messages
            error_messages = [r.get("error", "") for r in error_results]

            # Calculate average tokens per request
            avg_tokens_per_request = (
                sum(r.get("tokens", 0) for r in successful_results)
                / len(successful_results)
                if successful_results
                else 0
            )

            result = BenchmarkResult(
                model_name=model_config["name"],
                model_id=model_config["id"],
                concurrency=request.concurrency,
                prompt=request.prompt,
                total_time=raw_results["total_time"],
                avg_time_to_first_token=raw_results["avg_time_to_first_token"],
                avg_tokens_per_second=raw_results["avg_tokens_per_second"],
                aggregate_tokens_per_second=raw_results["aggregate_tokens_per_second"],
                peak_tokens_per_second=peak_tps,
                total_requests=raw_results["total_requests"],
                successful_requests=raw_results["successful_requests"],
                error_rate=raw_results["error_rate"],
                total_tokens_generated=raw_results["total_tokens"],
                avg_tokens_per_request=avg_tokens_per_request,
                sample_completion=raw_results.get("sample_completion", ""),
                completion_lengths=completion_lengths,
                individual_results=individual_results,
                error_messages=error_messages,
                timestamp=start_time,
                config_used={
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "model_config": model_config,
                },
            )

            if progress_callback:
                progress_callback(
                    "completed",
                    {
                        "model": model_config["name"],
                        "results": result.to_dict(),
                        "status": "Benchmark completed successfully",
                    },
                )

            return result

        except Exception as e:
            error_msg = f"Benchmark failed: {str(e)}"
            logger.error(error_msg)

            if progress_callback:
                progress_callback(
                    "error",
                    {
                        "model": model_config["name"],
                        "error": error_msg,
                        "status": "Benchmark failed",
                    },
                )

            # Return empty result with error information
            return BenchmarkResult(
                model_name=model_config["name"],
                model_id=model_config["id"],
                concurrency=request.concurrency,
                prompt=request.prompt,
                total_time=time.time() - start_time,
                avg_time_to_first_token=0,
                avg_tokens_per_second=0,
                aggregate_tokens_per_second=0,
                peak_tokens_per_second=0,
                total_requests=request.concurrency,
                successful_requests=0,
                error_rate=1.0,
                total_tokens_generated=0,
                avg_tokens_per_request=0,
                sample_completion="",
                completion_lengths=[],
                individual_results=[],
                error_messages=[error_msg],
                timestamp=start_time,
                config_used={
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "model_config": model_config,
                },
            )

    async def run_comparison_benchmark(
        self,
        model_keys: List[str],
        prompt: str,
        concurrency: int = 10,
        max_tokens: int = 256,
        temperature: float = 0.7,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark comparison across multiple models

        Args:
            model_keys: List of model keys to compare
            prompt: Test prompt
            concurrency: Number of concurrent requests per model
            temperature: Sampling temperature
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping model keys to their benchmark results
        """
        results = {}

        if progress_callback:
            progress_callback(
                "starting_comparison",
                {
                    "models": [
                        self.config.get_model(key)["name"] for key in model_keys
                    ],
                    "status": "Starting comparison benchmark...",
                },
            )

        # Run benchmarks for each model
        for i, model_key in enumerate(model_keys):
            if progress_callback:
                progress_callback(
                    "model_progress",
                    {
                        "current_model": self.config.get_model(model_key)["name"],
                        "progress": i / len(model_keys),
                        "status": f"Testing model {i + 1} of {len(model_keys)}",
                    },
                )

            request = BenchmarkRequest(
                model_key=model_key,
                prompt=prompt,
                concurrency=concurrency,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            results[model_key] = await self.run_single_benchmark(
                request, progress_callback
            )

        if progress_callback:
            progress_callback(
                "comparison_completed",
                {
                    "results": {k: v.to_dict() for k, v in results.items()},
                    "status": "Comparison benchmark completed",
                },
            )

        return results

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models with their configurations"""
        return self.config.get_all_models()

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        return self.config.get_model(model_key)


class BenchmarkReporter:
    """Generate reports from benchmark results"""

    @staticmethod
    def generate_comparison_report(
        results: Dict[str, BenchmarkResult],
    ) -> Dict[str, Any]:
        """Generate a comparison report from multiple benchmark results"""
        if not results:
            return {"error": "No results to compare"}

        # Find best performers
        best_aggregate_tps = max(
            results.values(), key=lambda r: r.aggregate_tokens_per_second
        )
        best_avg_tps = max(results.values(), key=lambda r: r.avg_tokens_per_second)
        best_ttft = min(results.values(), key=lambda r: r.avg_time_to_first_token)
        most_reliable = max(results.values(), key=lambda r: r.success_rate)

        return {
            "summary": {
                "models_tested": len(results),
                "total_requests": sum(r.total_requests for r in results.values()),
                "total_successful": sum(
                    r.successful_requests for r in results.values()
                ),
            },
            "winners": {
                "best_aggregate_throughput": {
                    "model": best_aggregate_tps.model_name,
                    "tokens_per_second": best_aggregate_tps.aggregate_tokens_per_second,
                },
                "best_average_throughput": {
                    "model": best_avg_tps.model_name,
                    "tokens_per_second": best_avg_tps.avg_tokens_per_second,
                },
                "fastest_first_token": {
                    "model": best_ttft.model_name,
                    "time_ms": best_ttft.avg_time_to_first_token * 1000,
                },
                "most_reliable": {
                    "model": most_reliable.model_name,
                    "success_rate": most_reliable.success_rate,
                },
            },
            "detailed_results": {k: v.to_dict() for k, v in results.items()},
            "generated_at": time.time(),
        }

    @staticmethod
    def export_to_json(results: Dict[str, BenchmarkResult], filepath: str) -> None:
        """Export benchmark results to JSON file"""
        report = BenchmarkReporter.generate_comparison_report(results)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    @staticmethod
    def export_to_csv(results: Dict[str, BenchmarkResult], filepath: str) -> None:
        """Export benchmark results to CSV file"""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Model",
                    "Model_ID",
                    "Concurrency",
                    "Total_Requests",
                    "Successful_Requests",
                    "Success_Rate_%",
                    "Total_Time_s",
                    "Avg_TPS",
                    "Aggregate_TPS",
                    "Peak_TPS",
                    "Avg_TTFT_ms",
                    "Total_Tokens",
                    "Avg_Tokens_Per_Request",
                    "Error_Rate_%",
                ]
            )

            # Data rows
            for result in results.values():
                writer.writerow(
                    [
                        result.model_name,
                        result.model_id,
                        result.concurrency,
                        result.total_requests,
                        result.successful_requests,
                        result.success_rate,
                        result.total_time,
                        result.avg_tokens_per_second,
                        result.aggregate_tokens_per_second,
                        result.peak_tokens_per_second,
                        result.avg_time_to_first_token * 1000,
                        result.total_tokens_generated,
                        result.avg_tokens_per_request,
                        result.error_rate * 100,
                    ]
                )


# Example usage and testing
async def example_usage():
    """Example of how to use the benchmark service"""
    # Initialize service
    service = FireworksBenchmarkService(api_key="your-api-key")

    # Single model benchmark
    request = BenchmarkRequest(
        model_key="qwen3_235b",
        prompt="Tell me a story about a brave knight.",
        concurrency=5,
        max_tokens=200,
    )

    result = await service.run_single_benchmark(request)
    print(f"Benchmark completed: {result.avg_tokens_per_second:.2f} TPS")

    # Comparison benchmark
    comparison_results = await service.run_comparison_benchmark(
        model_keys=["qwen3_235b", "llama_scout"],
        prompt="Explain quantum computing in simple terms.",
        concurrency=10,
    )

    # Generate report
    report = BenchmarkReporter.generate_comparison_report(comparison_results)
    print(f"Best model: {report['winners']['best_aggregate_throughput']['model']}")


if __name__ == "__main__":
    # For testing
    asyncio.run(example_usage())
