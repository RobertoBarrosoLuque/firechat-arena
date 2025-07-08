import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from fireworks import LLM
import yaml
from pathlib import Path
from src.logger import logger
from dotenv import load_dotenv

load_dotenv()

_FILE_PATH = Path(__file__).parents[2]
_CONFIG_PATH = _FILE_PATH / "configs" / "config.yaml"


@dataclass
class StreamingStats:
    """Statistics for a streaming response"""

    request_id: str
    start_time: float
    first_token_time: Optional[float] = None
    tokens_generated: int = 0
    characters_generated: int = 0
    completion_text: str = ""
    error_message: Optional[str] = None

    @property
    def total_time(self) -> float:
        """Total time in seconds"""
        return time.time() - self.start_time

    @property
    def time_to_first_token(self) -> float:
        """Time to first token in seconds"""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return 0

    @property
    def tokens_per_second(self) -> float:
        """Tokens per second throughput"""
        if self.total_time > 0 and self.tokens_generated > 0:
            return self.tokens_generated / self.total_time
        return 0


class FireworksConfig:
    """Configuration loader for Fireworks models"""

    def __init__(self):
        with open(_CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

    def get_model(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration by key"""
        if model_key not in self.config["models"]:
            raise ValueError(f"Model {model_key} not found in config")
        return self.config["models"][model_key]

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models"""
        return self.config["models"]

    def get_defaults(self) -> Dict[str, Any]:
        """Get default settings"""
        return self.config.get("defaults", {})


class FireworksStreamer:
    """Helper class for streaming responses from Fireworks"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.config = FireworksConfig()
        self._llm_cache = {}

    def _get_llm(self, model_key: str) -> LLM:
        """Get or create LLM instance for model"""
        if model_key not in self._llm_cache:
            model_config = self.config.get_model(model_key)
            self._llm_cache[model_key] = LLM(
                model=model_config["id"],
                deployment_type=model_config.get("deployment_type", "serverless"),
                api_key=self.api_key,
            )
        return self._llm_cache[model_key]

    def _prepare_llm_request(
        self,
        request_id: Optional[str],
        temperature: Optional[float],
        request_prefix: str,
    ) -> Tuple[str, float, StreamingStats]:
        """Prepares parameters and stats for an LLM request."""
        if not request_id:
            request_id = f"{request_prefix}_{int(time.time() * 1000)}"

        defaults = self.config.get_defaults()

        temperature = temperature or defaults.get("temperature", 0.7)

        stats = StreamingStats(request_id=request_id, start_time=time.time())

        return request_id, temperature, stats

    async def stream_completion(
        self,
        model_key: str,
        prompt: str,
        request_id: str = None,
        temperature: float = None,
        callback: Optional[Callable[[str, StreamingStats], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from Fireworks model

        Args:
            model_key: Key for model in config
            prompt: Input prompt
            request_id: Unique request identifier
            temperature: Sampling temperature
            callback: Optional callback for streaming stats

        Yields:
            Text chunks as they're generated
        """
        request_id, temperature, stats = self._prepare_llm_request(
            request_id, temperature, "req"
        )

        try:
            llm = self._get_llm(model_key)

            # Create streaming completion
            response_generator = llm.completions.create(
                prompt=prompt,
                temperature=temperature,
                stream=True,
            )

            async for chunk in self._async_generator_wrapper(response_generator):
                if chunk.choices and len(chunk.choices) > 0:
                    text = chunk.choices[0].text or ""

                    if text:
                        # Update stats
                        if stats.first_token_time is None:
                            stats.first_token_time = time.time()

                        stats.completion_text += text
                        stats.characters_generated += len(text)
                        # Rough token estimation (adjust as needed)
                        stats.tokens_generated = len(stats.completion_text.split())

                        # Call callback if provided
                        if callback:
                            callback(text, stats)

                        yield text

        except Exception as e:
            error_msg = f"Error in streaming completion: {str(e)}"
            logger.error(error_msg)
            stats.error_message = error_msg

            if callback:
                callback("", stats)

            raise

    async def stream_chat_completion(
        self,
        model_key: str,
        messages: list,
        request_id: str = None,
        temperature: float = None,
        callback: Optional[Callable[[str, StreamingStats], None]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion from Fireworks model

        Args:
            model_key: Key for model in config
            messages: List of chat messages
            request_id: Unique request identifier
            temperature: Sampling temperature
            callback: Optional callback for streaming stats

        Yields:
            Text chunks as they're generated
        """
        request_id, temperature, stats = self._prepare_llm_request(
            request_id, temperature, "chat"
        )

        try:
            llm = self._get_llm(model_key)

            # Create streaming chat completion
            response_generator = llm.chat.completions.create(
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            async for chunk in self._async_generator_wrapper(response_generator):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    text = delta.content if delta else ""

                    if text:
                        # Update stats
                        if stats.first_token_time is None:
                            stats.first_token_time = time.time()

                        stats.completion_text += text
                        stats.characters_generated += len(text)
                        # Rough token estimation
                        stats.tokens_generated = len(stats.completion_text.split())

                        # Call callback if provided
                        if callback:
                            callback(text, stats)

                        yield text

        except Exception as e:
            error_msg = f"Error in streaming chat completion: {str(e)}"
            logger.error(error_msg)
            stats.error_message = error_msg

            if callback:
                callback("", stats)

            raise

    async def _async_generator_wrapper(self, sync_generator):
        """Convert sync generator to async generator"""
        loop = asyncio.get_event_loop()

        def get_next():
            try:
                return next(sync_generator)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, get_next)
            if chunk is None:
                break
            yield chunk
            # Allow other tasks to run
            await asyncio.sleep(0)


class FireworksBenchmark:
    """Benchmark helper for Fireworks models"""

    def __init__(self, api_key: str):
        self.streamer = FireworksStreamer(api_key)
        self.config = FireworksConfig()

    async def run_concurrent_benchmark(
        self,
        model_key: str,
        prompt: str,
        concurrency: int = 10,
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        Run concurrent requests for benchmarking

        Returns:
            Dictionary with aggregated benchmark results
        """
        start_time = time.time()

        async def single_request(req_id: int):
            request_stats = []

            def stats_callback(text: str, stats: StreamingStats):
                request_stats.append(
                    {
                        "time": stats.total_time,
                        "tokens": stats.tokens_generated,
                        "ttft": stats.time_to_first_token,
                        "tps": stats.tokens_per_second,
                    }
                )

            try:
                completion_text = ""
                async for chunk in self.streamer.stream_completion(
                    model_key=model_key,
                    prompt=prompt,
                    request_id=f"bench_{req_id}",
                    temperature=temperature,
                    callback=stats_callback,
                ):
                    completion_text += chunk

                final_stats = (
                    request_stats[-1]
                    if request_stats
                    else {"time": 0, "tokens": 0, "ttft": 0, "tps": 0}
                )
                final_stats["completion_text"] = completion_text
                final_stats["request_id"] = req_id
                return final_stats

            except Exception as e:
                logger.error(f"Request {req_id} failed: {str(e)}")
                return {
                    "request_id": req_id,
                    "time": 0,
                    "tokens": 0,
                    "ttft": 0,
                    "tps": 0,
                    "error": str(e),
                    "completion_text": "",
                }

        # Run concurrent requests
        tasks = [single_request(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            r
            for r in results
            if not isinstance(r, Exception) and r.get("tokens", 0) > 0
        ]

        total_time = time.time() - start_time

        if not successful_results:
            return {
                "model": self.config.get_model(model_key)["name"],
                "concurrency": concurrency,
                "total_requests": concurrency,
                "successful_requests": 0,
                "total_time": total_time,
                "avg_tokens_per_second": 0,
                "avg_time_to_first_token": 0,
                "aggregate_tokens_per_second": 0,
                "total_tokens": 0,
                "error_rate": 1.0,
            }

        # Calculate aggregated metrics
        total_tokens = sum(r["tokens"] for r in successful_results)
        avg_tps = sum(r["tps"] for r in successful_results) / len(successful_results)
        avg_ttft = sum(r["ttft"] for r in successful_results) / len(successful_results)
        aggregate_tps = total_tokens / total_time if total_time > 0 else 0

        return {
            "model": self.config.get_model(model_key)["name"],
            "concurrency": concurrency,
            "total_requests": concurrency,
            "successful_requests": len(successful_results),
            "total_time": total_time,
            "avg_tokens_per_second": avg_tps,
            "avg_time_to_first_token": avg_ttft,
            "aggregate_tokens_per_second": aggregate_tps,
            "total_tokens": total_tokens,
            "error_rate": (concurrency - len(successful_results)) / concurrency,
            "sample_completion": (
                successful_results[0]["completion_text"] if successful_results else ""
            ),
            "individual_results": successful_results,
        }
