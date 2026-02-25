import argparse
from pathlib import Path

from floeval.api.dataset import DatasetLoader
from floeval.cli import CLIGenerationConfig
from floeval.cli.export import export_dataset_to_json, export_dataset_to_jsonl
from floeval.cli.utils import CLIConfigLoader, check_if_file_exists
from floeval.config.schemas.io.dataset import Dataset, PartialDataset
from floeval.core.execution.llm_executor import OpenAIProvider
from floeval.core.execution.response_synthesizer import populate_llm_responses


def parse_args(args: argparse.Namespace):
    """Parse command-line arguments for the 'generate' subcommand.

    Args:
        args: The command-line arguments to parse.
    """
    partial_dataset_file = Path(args.dataset) if args.dataset else None
    output_file = Path(args.output) if args.output else None
    config_file = Path(args.config) if args.config else None

    assert (partial_dataset_file is not None, 
    ), "Dataset file path must be provided with --dataset"
    assert config_file is not None, "Config file path must be provided with --config"
    assert output_file is not None, "Output file path must be provided with --output"

    if output_file and not output_file.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_file.parent}; please provide a valid output path with --output"
        )

    partial_dataset_file_path = check_if_file_exists(partial_dataset_file)
    config_file_path = check_if_file_exists(config_file)

    partial_dataset = DatasetLoader.from_file(
        partial_dataset_file_path, partial_dataset=True
    )
    assert partial_dataset is not None, "Failed to load partial dataset from file"
    assert isinstance(
        partial_dataset, PartialDataset
    ), f"Expected a PartialDataset instance; got {type(partial_dataset)}"

    config_loader = CLIConfigLoader(model_class=CLIGenerationConfig)
    generation_config = config_loader.load(config_file_path)

    _model_config_name = generation_config.dataset_generation_config.generator_model
    batch_size = generation_config.dataset_generation_config.batch_size
    max_concurrency = generation_config.dataset_generation_config.max_concurrency
    llm_provider = OpenAIProvider(
        config_name=f"{_model_config_name}_generation",
        **generation_config.llm_config | {"chat_model": _model_config_name},
    )

    dataset: Dataset = populate_llm_responses(
        partial_dataset=partial_dataset,
        llm_provider=llm_provider,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )

    match output_file.suffix.lower():
        case ".json":
            export_dataset_to_json(dataset, output_file)
            print(f"Dataset successfully saved to {output_file}")
        case ".jsonl":
            export_dataset_to_jsonl(dataset, output_file)
            print(f"Dataset successfully saved to {output_file}")
        case _:
            print(
                f"Unsupported output file format: {output_file.suffix}. "
                "Please provide a .json or .jsonl file extension."
            )
