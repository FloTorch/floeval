import asyncio

from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    PartialSample,
    Sample,
    convert_partial_to_full_sample,
)
from floeval.config.schemas.prompts import PromptFile
from floeval.core.execution.base import BaseLLMProvider
from floeval.utils.asyncio_compat import run_coroutine_sync

GenerationWorkItem = tuple[int, PartialSample, str | None, str | None]


def _resolve_system_prompt(
    prompt_id: str | None,
    prompts: PromptFile | None,
) -> str | None:
    if not prompts or not prompt_id:
        return None
    prompt = prompts.prompts.get(prompt_id)
    return prompt.template if prompt else None


def _build_generation_work_items(
    partial_dataset: PartialDataset,
    prompts: PromptFile | None,
) -> tuple[list[Sample | None], list[GenerationWorkItem]]:
    ordered_samples: list[Sample | None] = []
    generation_work_items: list[GenerationWorkItem] = []

    for partial_sample in partial_dataset.samples:
        if partial_sample.llm_response:
            sample_data = partial_sample.model_dump(exclude={"prompt_ids"})
            ordered_samples.append(Sample(**sample_data))
            continue

        prompt_ids_to_process = (
            partial_sample.prompt_ids if partial_sample.prompt_ids else [None]
        )
        for prompt_id in prompt_ids_to_process:
            result_index = len(ordered_samples)
            ordered_samples.append(None)
            generation_work_items.append(
                (
                    result_index,
                    partial_sample,
                    prompt_id,
                    _resolve_system_prompt(prompt_id=prompt_id, prompts=prompts),
                )
            )

    return ordered_samples, generation_work_items


async def _generate_single_response(
    item: GenerationWorkItem,
    llm_provider: BaseLLMProvider,
    semaphore: asyncio.Semaphore,
) -> tuple[int, Sample]:
    result_index, partial_sample, prompt_id, system_prompt = item
    async with semaphore:
        llm_response = await llm_provider.agenerate(
            partial_sample.user_input,
            system_prompt=system_prompt,
        )
    return (
        result_index,
        convert_partial_to_full_sample(
            partial_sample=partial_sample,
            llm_response=llm_response,
            prompt_id=prompt_id,
        ),
    )


async def _process_generation_batch(
    batch: list[GenerationWorkItem],
    llm_provider: BaseLLMProvider,
    semaphore: asyncio.Semaphore,
) -> list[tuple[int, Sample]]:
    tasks = [
        _generate_single_response(
            item=item,
            llm_provider=llm_provider,
            semaphore=semaphore,
        )
        for item in batch
    ]
    return await asyncio.gather(*tasks)


async def apopulate_llm_responses(
    partial_dataset: PartialDataset,
    llm_provider: BaseLLMProvider,
    prompts: PromptFile | None = None,
    batch_size: int = 20,
    max_concurrency: int = 10,
) -> Dataset:
    """Populate missing LLM responses asynchronously.

    This is the primary API for async callers, including notebooks with
    top-level ``await``.

    Args:
        partial_dataset: Dataset with missing llm_response fields.
        llm_provider: LLM provider for response generation.
        prompts: Optional prompt definitions for system_prompt injection.
        batch_size: Number of requests to process per batch.
        max_concurrency: Max concurrent requests inside each batch.

    Returns:
        Dataset: A dataset with llm_response filled for all samples.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than 0")

    ordered_samples, generation_work_items = _build_generation_work_items(
        partial_dataset=partial_dataset,
        prompts=prompts,
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    for batch_start in range(0, len(generation_work_items), batch_size):
        batch = generation_work_items[batch_start : batch_start + batch_size]
        batch_results = await _process_generation_batch(
            batch=batch,
            llm_provider=llm_provider,
            semaphore=semaphore,
        )
        for result_index, sample in batch_results:
            ordered_samples[result_index] = sample

    # all placeholders are guaranteed filled if task execution succeeds
    return Dataset(samples=[sample for sample in ordered_samples if sample is not None])


def populate_llm_responses(
    partial_dataset: PartialDataset,
    llm_provider: BaseLLMProvider,
    prompts: PromptFile | None = None,
    batch_size: int = 20,
    max_concurrency: int = 10,
) -> Dataset:
    """Generate LLM responses, optionally using prompt templates.

    This is a synchronous convenience wrapper around
    ``apopulate_llm_responses``.

    If prompt_ids is provided on a sample, generates one response per prompt_id,
    expanding the dataset accordingly. If prompt_ids is not provided, generates
    a single response without system prompt injection (current behavior).

    Note:
        In async code or notebook cells with a running event loop, prefer
        ``await apopulate_llm_responses(...)`` to avoid sync-over-async
        bridging overhead.

    Args:
        partial_dataset: Dataset with missing llm_response fields
        llm_provider: LLM provider for response generation
        prompts: Optional prompt definitions for system_prompt injection
        batch_size: Number of requests to process per batch
        max_concurrency: Max concurrent requests inside each batch

    Returns:
        Dataset: A dataset with llm_response field filled in all samples
    """
    return run_coroutine_sync(
        lambda: apopulate_llm_responses(
            partial_dataset=partial_dataset,
            llm_provider=llm_provider,
            prompts=prompts,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
    )
