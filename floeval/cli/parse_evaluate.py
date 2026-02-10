import argparse
import json
from pathlib import Path

from floeval.api.dataset import DatasetLoader
from floeval.api.evaluation import Evaluation, EvaluationResult
from floeval.cli import ConfigError
from floeval.cli.utils import EvalConfigLoader
from floeval.config import GatewayConfig


def check_if_file_exists(file_path: str) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def _pretty_print_results(results: EvaluationResult):
    print("\n" + "-" * 80)

    print("Detailed Results:")
    print("-" * 80)
    for i, sr in enumerate(results.sample_results, start=1):
        print(f"\nSample {i}: {sr['user_input']}")
        for metric_key, metric_data in sr["metrics"].items():
            score = metric_data["score"]
            passed = metric_data.get("passed")
            err = metric_data["metadata"].get("error")

            if score is None:
                print(f"  ❌ {metric_key}: FAILED")
                if err:
                    print(f"     Error: {err}")
            else:
                status = "PASSED" if passed else "WARNING"
                print(f"  {status} {metric_key}: {score:.4f}")
                if err:
                    print(f"     Warning: {err}")


def output_results(results: EvaluationResult, output_path: Path | None):
    """Utility function to process evaluation results. Writes to a JSON file if output_path is provided and valid.
    If output_path is not provided: writes summary of evaluation results to the console.

    Args:
        results (EvaluationResult): The evaluation results to be processed.
        output_path (str | None): The path to save the results. If None, results will be printed to the console.
    """
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(results.model_dump(), f, indent=4)
            print(f"Results successfully saved to {output_path}")
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")
    else:
        _pretty_print_results(results)


def parse_args(args: argparse.Namespace):
    config_file = args.config
    output_file = Path(args.output) if args.output else None

    if output_file and not output_file.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_file.parent}; please provide a valid output path with --output"
        )

    try:
        dataset_file = check_if_file_exists(args.dataset)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset file error: {e}; please provide a valid dataset file path with --dataset"
        ) from e

    # ----- Load evaluation configuration (YAML or JSON) -----
    config_loader = EvalConfigLoader()
    config_data = config_loader.load(config_file)
    gateway_config_data = config_data.get("gateway_config", {})
    if not gateway_config_data:
        raise ConfigError("Missing 'gateway_config' section in the configuration file")
    eval_config = config_data.get("evaluation_config", {})
    if not eval_config:
        raise ConfigError(
            "Missing 'evaluation_config' section in the configuration file"
        )
    gateway_config = GatewayConfig(
        gateway_base_url=gateway_config_data["floeval_gateway_base_url"],
        api_key=gateway_config_data["llm_api_key"],
        llm_model=gateway_config_data["llm_model_name"],
        embedding_model=gateway_config_data["embedding_model_name"],
    )

    dataset = DatasetLoader.from_file(dataset_file)
    evaluation = Evaluation(
        dataset=dataset,
        gateway_config=gateway_config,
        default_provider=eval_config.get("default_provider"),
        metrics=eval_config["metrics"],
        metric_params=eval_config.get("metric_params", {}),
    )
    results = evaluation.run()

    output_results(results, output_file)
