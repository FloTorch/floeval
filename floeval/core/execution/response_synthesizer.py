from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    Sample,
    convert_partial_to_full_sample,
)
from floeval.config.schemas.prompts import PromptFile
from floeval.core.execution.base import BaseLLMProvider


def populate_llm_responses(
    partial_dataset: PartialDataset,
    llm_provider: BaseLLMProvider,
    prompts: PromptFile | None = None,
) -> Dataset:
    """Generate LLM responses, optionally using prompt templates.

    If prompt_ids is provided on a sample, generates one response per prompt_id,
    expanding the dataset accordingly. If prompt_ids is not provided, generates
    a single response without system prompt injection (current behavior).

    Args:
        partial_dataset: Dataset with missing llm_response fields
        llm_provider: LLM provider for response generation
        prompts: Optional prompt definitions for system_prompt injection

    Returns:
        Dataset: A dataset with llm_response field filled in all samples
    """
    _samples: list[Sample] = []

    for partial_sample in partial_dataset.samples:
        if partial_sample.llm_response:
            # Already has response, keep as-is (exclude prompt_ids from Sample)
            sample_data = partial_sample.model_dump(exclude={"prompt_ids"})
            _samples.append(Sample(**sample_data))
            continue

        # Determine which prompt_ids to process
        prompt_ids_to_process = (
            partial_sample.prompt_ids if partial_sample.prompt_ids else [None]
        )

        for prompt_id in prompt_ids_to_process:
            system_prompt = None
            if prompts and prompt_id:
                prompt = prompts.prompts.get(prompt_id)
                if prompt:
                    system_prompt = prompt.template

            _llm_response = llm_provider.generate(
                partial_sample.user_input,
                system_prompt=system_prompt,
            )

            _sample = convert_partial_to_full_sample(
                partial_sample=partial_sample,
                llm_response=_llm_response,
                prompt_id=prompt_id,
            )
            _samples.append(_sample)

    return Dataset(samples=_samples)
