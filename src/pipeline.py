# src/pipeline.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import litellm
from litellm import RateLimitError  # Import the RateLimitError
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random
from tqdm import tqdm

import utils

logger = utils.configure_logger()


@dataclass
class PipelineConfig:
    """Configuration for LLM Pipeline"""
    model: str = "gpt-4.1-nano"
    timeout: int = 30
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    max_workers: int = 8
    prompt_path: str = ""
    num_retries: int = 5


class StructuredLLMPipeline:
    """
    A pipeline for processing data through LLMs with structured outputs.

    Features:
    - Loads prompts from version-controlled files
    - Uses Pydantic models for output validation
    - Implements retry logic for API calls
    - Controls concurrent requests
    - Shows progress with tqdm
    """

    def __init__(
        self,
        config: PipelineConfig,
        output_schema: Type[BaseModel],
    ):
        self.prompt_template = Path(config.prompt_path).read_text()
        self.output_schema = output_schema
        self.config = config

        logger.info(f"Initialized pipeline with model={self.config.model}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=30, max=600) + wait_random(0, 30),
    )
    def _single_completion(self, message: List[dict]) -> Optional[BaseModel]:
        """Make a single API call with retry logic"""
        try:
            response = litellm.completion(
                model=self.config.model,
                messages=message,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                response_format=self.output_schema,
                num_retries=self.config.num_retries,
            )
            return self.output_schema.model_validate_json(response.choices[0].message.content)
        except RateLimitError as rate_limit_err:
            logger.debug(f"Rate limit exceeded during completion: {str(rate_limit_err)}. Retrying...")
            # You may want to return None or raise to trigger retry
            raise rate_limit_err  # This will allow the retry logic to kick in

        except Exception as e:
            logger.error(f"Error in single completion: {str(e)}")
            raise

    def _batch_process(
        self,
        messages: List[List[dict]],
        batch_size: int = 10,
    ) -> List[Optional[BaseModel]]:
        """Process messages in parallel with controlled concurrency and progress tracking"""
        results = [None] * len(messages)
        failed_indices = []

        # Create progress bar for batches
        with tqdm(total=len(messages), desc="Processing items") as pbar:
            for batch_start in range(0, len(messages), batch_size):
                batch_end = min(batch_start + batch_size, len(messages))
                batch = messages[batch_start:batch_end]

                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_index = {
                        executor.submit(self._single_completion, message): idx + batch_start
                        for idx, message in enumerate(batch)
                    }

                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            result = future.result()
                            results[idx] = result
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Failed processing item {idx}: {str(e)}")
                            failed_indices.append(idx)
                            pbar.update(1)

        # Retry failed requests
        if failed_indices:
            logger.info(f"Retrying {len(failed_indices)} failed requests...")
            with tqdm(total=len(failed_indices), desc="Retrying failed items") as pbar:
                for idx in failed_indices:
                    try:
                        results[idx] = self._single_completion(messages[idx])
                    except Exception as e:
                        logger.error(f"Final retry failed for item {idx}: {str(e)}")
                        results[idx] = None
                    pbar.update(1)

        return results

    def process_items(self, items: List[Dict[str, Any]]) -> List[Optional[BaseModel]]:
        """Process a list of items through the LLM pipeline."""
        if not items:
            logger.warning("Received empty items list")
            return []

        messages = [
            [{"role": "user", "content": self.prompt_template.format(**item)}]
            for item in items
        ]

        try:
            results = self._batch_process(messages)
            successful = sum(1 for r in results if r is not None)
            logger.info(
                f"Processing complete: {successful}/{len(items)} successful "
                f"({successful / len(items) * 100:.1f}%)"
            )
            return results

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return [None] * len(messages)