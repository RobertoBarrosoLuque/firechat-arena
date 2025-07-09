from modules.benchmark import BenchmarkResult


def test_rps_calculation():
    """Test that RPS is calculated correctly"""

    # Create a mock benchmark result
    result = BenchmarkResult(
        model_name="Test Model",
        model_id="test_model",
        concurrency=10,
        prompt="Test prompt",
        total_time=5.0,  # 5 seconds
        avg_time_to_first_token=0.5,
        avg_tokens_per_second=50.0,
        aggregate_tokens_per_second=500.0,
        peak_tokens_per_second=80.0,
        total_requests=20,  # Total requests attempted
        successful_requests=18,  # 18 successful requests
        error_rate=0.1,  # 10% error rate
        total_tokens_generated=2500,
        avg_tokens_per_request=138.9,
        sample_completion="This is a test completion",
        completion_lengths=[100, 150, 120, 140, 130],
        individual_results=[],
        error_messages=[],
        timestamp=1234567890.0,
        config_used={},
    )

    # Test RPS calculation
    expected_rps = 18 / 5.0  # 18 successful requests / 5 seconds = 3.6 RPS
    actual_rps = result.requests_per_second

    assert (
        abs(actual_rps - expected_rps) < 0.001
    ), f"Expected RPS {expected_rps}, got {actual_rps}"


def test_zero_time_edge_case():
    """Test RPS calculation when total_time is zero"""

    result = BenchmarkResult(
        model_name="Test Model",
        model_id="test_model",
        concurrency=10,
        prompt="Test prompt",
        total_time=0.0,  # Zero time
        avg_time_to_first_token=0.0,
        avg_tokens_per_second=0.0,
        aggregate_tokens_per_second=0.0,
        peak_tokens_per_second=0.0,
        total_requests=10,
        successful_requests=10,
        error_rate=0.0,
        total_tokens_generated=0,
        avg_tokens_per_request=0.0,
        sample_completion="",
        completion_lengths=[],
        individual_results=[],
        error_messages=[],
        timestamp=1234567890.0,
        config_used={},
    )

    actual_rps = result.requests_per_second
    assert actual_rps == 0.0, f"Zero time should return 0 RPS, got {actual_rps}"


def test_realistic_benchmark_scenario():
    """Test with realistic benchmark numbers"""

    # Simulate a 10-second benchmark with 50 concurrent requests
    result = BenchmarkResult(
        model_name="Qwen3 235B",
        model_id="qwen3_235b",
        concurrency=50,
        prompt="What is machine learning?",
        total_time=10.5,  # 10.5 seconds
        avg_time_to_first_token=1.2,  # 1.2 seconds TTFT
        avg_tokens_per_second=45.5,  # 45.5 TPS per request
        aggregate_tokens_per_second=2275.0,  # Total throughput
        peak_tokens_per_second=78.2,
        total_requests=50,
        successful_requests=48,  # 2 failed requests
        error_rate=0.04,  # 4% error rate
        total_tokens_generated=23887,
        avg_tokens_per_request=497.6,
        sample_completion="Machine learning is a subset of artificial intelligence...",
        completion_lengths=[480, 520, 475, 510, 490],
        individual_results=[],
        error_messages=["Timeout", "Rate limit"],
        timestamp=1234567890.0,
        config_used={"temperature": 0.7, "max_tokens": 500},
    )

    expected_rps = 48 / 10.5  # ~4.57 RPS
    actual_rps = result.requests_per_second

    # Verify RPS is much lower than TPS (as expected)
    assert (
        abs(actual_rps - expected_rps) < 0.01
    ), f"Expected RPS {expected_rps:.2f}, got {actual_rps:.2f}"
    assert (
        actual_rps < result.avg_tokens_per_second
    ), "RPS should be much lower than TPS"
    assert (
        actual_rps < result.aggregate_tokens_per_second
    ), "RPS should be much lower than aggregate TPS"
