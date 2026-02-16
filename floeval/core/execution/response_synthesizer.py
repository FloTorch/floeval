from floeval.config.schemas.io.dataset import (
    Dataset,
    PartialDataset,
    Sample,
    convert_partial_to_full_sample,
)
from floeval.core.execution.llm_executor import BaseLLMProvider


def populate_llm_responses(
    partial_dataset: PartialDataset, llm_provider: BaseLLMProvider
) -> Dataset:
    """Method to generate llm responses by generating response from llm using llm_provider instance.

    Args:
        partial_dataset: partially filled dataset - missing llm_response field in the samples
        llm_provider: an instance of BaseLLMProvider that can be used to generate llm responses

    Returns:
        Dataset: a dataset with llm_response field filled in all samples
    """
    _samples = []
    for partial_sample in partial_dataset.samples:
        if not partial_sample.llm_response:
            _llm_response = llm_provider.generate(partial_sample.user_input)
            _sample = convert_partial_to_full_sample(partial_sample, _llm_response)
            _samples.append(_sample)
        else:
            _samples.append(Sample(**partial_sample.model_dump()))

    return Dataset(samples=_samples)
