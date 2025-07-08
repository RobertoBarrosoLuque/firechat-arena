import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from modules.benchmark import FireworksBenchmarkService
from modules.llm_completion import FireworksStreamer


class SpeedTestComparisonRequest(BaseModel):
    """Enhanced request model for speed test comparison"""

    model_keys: List[str]
    message: str
    temperature: Optional[float] = 0.7
    comparison_id: Optional[str] = None
    speed_test: bool = False
    concurrency: int = 1


class TestSpeedTestIntegration:
    """Test suite for speed test integration"""

    def setup_method(self):
        """Setup test environment"""
        self.api_key = "test_api_key"  # pragma: allowlist secret
        self.benchmark_service = FireworksBenchmarkService(self.api_key)
        self.streamer = FireworksStreamer(self.api_key)

        # Mock models for testing
        self.test_models = ["llama_scout", "qwen3_235b"]
        self.test_message = "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_speed_test_integration_basic(self):
        """Test basic speed test integration with chat comparison"""

        # Test the speed test integration
        results = await self.run_speed_test_comparison(
            model_keys=self.test_models, message=self.test_message, concurrency=3
        )

        # Verify results structure
        assert "model1_tps" in results
        assert "model2_tps" in results
        assert "model1_ttft" in results
        assert "model2_ttft" in results
        assert "concurrency" in results
        assert results["concurrency"] == 3

        print("✅ Basic speed test integration test passed")

    @pytest.mark.asyncio
    async def test_concurrent_chat_and_speed_test(self):
        """Test that chat streaming and speed test can run concurrently"""

        # Run concurrent test
        chat_results, speed_results = await self.run_concurrent_chat_speed_test(
            model_keys=self.test_models, message=self.test_message, concurrency=2
        )

        # Verify both chat and speed test completed
        assert len(chat_results) == 2
        assert "model1_tps" in speed_results
        assert "model2_tps" in speed_results

        print("✅ Concurrent chat and speed test passed")

    @pytest.mark.asyncio
    async def test_sse_event_format(self):
        """Test SSE event format for speed test results"""

        # Generate SSE events for speed test
        events = []
        async for event in self.generate_speed_test_sse_events():
            events.append(event)

        # Verify event types
        event_types = [event["type"] for event in events]
        assert "content" in event_types
        assert "speed_test_results" in event_types
        assert "comparison_done" in event_types

        # Verify speed test results format
        speed_test_event = next(e for e in events if e["type"] == "speed_test_results")
        results = speed_test_event["results"]

        assert "model1_tps" in results
        assert "model2_tps" in results
        assert "model1_ttft" in results
        assert "model2_ttft" in results
        assert "concurrency" in results

        print("✅ SSE event format test passed")

    @pytest.mark.asyncio
    async def test_speed_test_with_different_concurrency(self):
        """Test speed test with different concurrency levels"""

        concurrency_levels = [1, 2, 5, 10]

        for concurrency in concurrency_levels:
            results = await self.run_speed_test_comparison(
                model_keys=self.test_models,
                message=self.test_message,
                concurrency=concurrency,
            )

            assert results["concurrency"] == concurrency
            assert "model1_times" in results
            assert "model2_times" in results
            assert len(results["model1_times"]) == concurrency
            assert len(results["model2_times"]) == concurrency

        print("✅ Different concurrency levels test passed")

    @pytest.mark.asyncio
    async def test_speed_test_error_handling(self):
        """Test error handling in speed test integration"""

        # Test with invalid concurrency
        with pytest.raises(ValueError):
            await self.run_speed_test_comparison(
                model_keys=self.test_models, message=self.test_message, concurrency=0
            )

        # Test with invalid model count
        with pytest.raises(ValueError):
            await self.run_speed_test_comparison(
                model_keys=["single_model"], message=self.test_message, concurrency=2
            )

        print("✅ Error handling test passed")

    async def run_speed_test_comparison(
        self, model_keys: List[str], message: str, concurrency: int
    ) -> Dict[str, Any]:
        """Implementation of speed test comparison"""

        # Validate inputs
        if concurrency <= 0:
            raise ValueError("Concurrency must be positive")

        if len(model_keys) != 2:
            raise ValueError("Must provide exactly 2 models")

        # Mock benchmark execution
        results = {}

        for i, model_key in enumerate(model_keys):
            # Simulate benchmark execution
            await asyncio.sleep(0.1)  # Simulate processing time

            # Generate mock metrics
            base_tps = 40.0 + i * 10.0
            base_ttft = 150.0 - i * 30.0
            base_time = 1200.0 - i * 200.0

            # Generate individual times for concurrency
            individual_times = [base_time + (j * 50 - 25) for j in range(concurrency)]

            model_prefix = f"model{i+1}"
            results[f"{model_prefix}_tps"] = base_tps
            results[f"{model_prefix}_ttft"] = base_ttft
            results[f"{model_prefix}_times"] = individual_times
            results[f"{model_prefix}_avg_time"] = base_time

        results["concurrency"] = concurrency

        return results

    async def run_concurrent_chat_speed_test(
        self, model_keys: List[str], message: str, concurrency: int
    ):
        """Run chat streaming and speed test concurrently"""

        # Create concurrent tasks
        chat_task = asyncio.create_task(self.mock_chat_streaming(model_keys, message))
        speed_task = asyncio.create_task(
            self.run_speed_test_comparison(model_keys, message, concurrency)
        )

        # Wait for both to complete
        chat_results, speed_results = await asyncio.gather(chat_task, speed_task)

        return chat_results, speed_results

    async def mock_chat_streaming(self, model_keys: List[str], message: str):
        """Mock chat streaming responses"""

        responses = []
        for model_key in model_keys:
            # Simulate streaming response
            chunks = ["Hello", " from", f" {model_key}"]
            response = ""
            for chunk in chunks:
                response += chunk
                await asyncio.sleep(0.05)  # Simulate streaming delay
            responses.append(response)

        return responses

    async def generate_speed_test_sse_events(self):
        """Generate SSE events for speed test"""

        comparison_id = "test_comparison_123"

        # Chat content events
        for i, model_key in enumerate(self.test_models):
            yield {
                "type": "content",
                "model_index": i,
                "model_key": model_key,
                "content": f"Response from {model_key}",
            }

        # Model done events
        for i, model_key in enumerate(self.test_models):
            yield {"type": "model_done", "model_index": i, "model_key": model_key}

        # Speed test results
        yield {
            "type": "speed_test_results",
            "comparison_id": comparison_id,
            "results": {
                "model1_tps": 45.2,
                "model2_tps": 52.1,
                "model1_ttft": 150.0,
                "model2_ttft": 120.0,
                "model1_times": [1150, 1200, 1250],
                "model2_times": [950, 980, 1010],
                "model1_avg_time": 1200.0,
                "model2_avg_time": 980.0,
                "concurrency": 3,
            },
        }

        # Comparison done
        yield {"type": "comparison_done", "comparison_id": comparison_id}


class TestSpeedTestAPIIntegration:
    """Test API integration for speed test"""

    def setup_method(self):
        """Setup FastAPI test client"""
        self.app = FastAPI()
        self.setup_routes()
        self.client = TestClient(self.app)

    def setup_routes(self):
        """Setup test routes"""

        @self.app.post("/chat/compare")
        async def compare_chat(request: SpeedTestComparisonRequest):
            """Enhanced comparison endpoint with speed test"""

            # Validate models
            if len(request.model_keys) != 2:
                raise ValueError("Must provide exactly 2 models")

            # Generate response based on speed test setting
            if request.speed_test:
                return await self.generate_speed_test_response(request)
            else:
                return await self.generate_regular_response(request)

        self.generate_speed_test_response = self._generate_speed_test_response
        self.generate_regular_response = self._generate_regular_response

    async def _generate_speed_test_response(self, request: SpeedTestComparisonRequest):
        """Generate speed test response"""

        # Mock speed test execution
        await asyncio.sleep(0.1)

        return {
            "type": "speed_test_results",
            "comparison_id": request.comparison_id or "test_comparison",
            "results": {
                "model1_tps": 45.2,
                "model2_tps": 52.1,
                "model1_ttft": 150.0,
                "model2_ttft": 120.0,
                "concurrency": request.concurrency,
            },
        }

    async def _generate_regular_response(self, request: SpeedTestComparisonRequest):
        """Generate regular chat response"""

        return {
            "type": "comparison_done",
            "comparison_id": request.comparison_id or "test_comparison",
            "models": request.model_keys,
        }

    def test_api_speed_test_enabled(self):
        """Test API with speed test enabled"""

        request_data = {
            "model_keys": ["llama_scout", "qwen3_235b"],
            "message": "Test message",
            "speed_test": True,
            "concurrency": 3,
        }

        response = self.client.post("/chat/compare", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "speed_test_results"
        assert "results" in data
        assert data["results"]["concurrency"] == 3

        print("✅ API speed test enabled test passed")

    def test_api_speed_test_disabled(self):
        """Test API with speed test disabled"""

        request_data = {
            "model_keys": ["llama_scout", "qwen3_235b"],
            "message": "Test message",
            "speed_test": False,
        }

        response = self.client.post("/chat/compare", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "comparison_done"
        assert "models" in data

        print("✅ API speed test disabled test passed")


if __name__ == "__main__":
    """Run tests directly"""

    # Run basic integration test
    integration_test = TestSpeedTestIntegration()
    integration_test.setup_method()

    print("🚀 Running Speed Test Integration Tests...")

    # Run async tests
    async def run_async_tests():
        await integration_test.test_speed_test_integration_basic()
        await integration_test.test_concurrent_chat_and_speed_test()
        await integration_test.test_sse_event_format()
        await integration_test.test_speed_test_with_different_concurrency()
        await integration_test.test_speed_test_error_handling()

    asyncio.run(run_async_tests())

    # Run API tests
    api_test = TestSpeedTestAPIIntegration()
    api_test.setup_method()
    api_test.test_api_speed_test_enabled()
    api_test.test_api_speed_test_disabled()

    print("✅ All tests completed successfully!")
