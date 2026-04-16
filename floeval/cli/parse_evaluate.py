import argparse
import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from floeval.api.agent_evaluation import AgentEvaluation, AgentEvaluationResult
from floeval.api.dataset import DatasetLoader
from floeval.api.evaluation import Evaluation, EvaluationResult
from floeval.api.workflow_evaluation import WorkflowEvaluation
from floeval.cli import CLIEvaluationConfig, ConfigError
from floeval.cli.utils import CLIConfigLoader, check_if_file_exists
from floeval.config.schemas.io.agent_dataset import AgentDataset
from floeval.config.schemas.io.llm import LLMProviderConfig, OpenAIProviderConfig
from floeval.utils.asyncio_compat import run_coroutine_sync


def _ensure_output_parent(output_file: Path | None) -> None:
    if output_file and not output_file.parent.exists():
        raise FileNotFoundError(f"Output directory not found: {output_file.parent}")


def _resolve_dataset_path(dataset_arg: str) -> Path:
    try:
        return check_if_file_exists(dataset_arg)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {e}") from e


def _save_json_output(payload: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)
    print(f"Results saved to {output_path}")


def _is_partial_dataset(file_path: Path) -> bool:
    """Detect if the dataset file has samples missing llm_response (partial dataset)."""
    ext = file_path.suffix[1:].lower()
    if ext == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
    elif ext == "jsonl":
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    else:
        return False
    for s in samples:
        if "llm_response" not in s or s.get("llm_response") is None or s.get("llm_response") == "":
            return True
    return False


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
    """Process evaluation results & output them.

    Saves to JSON file if output_path is provided; otherwise prints to console.

    Args:
        results: The evaluation results to process.
        output_path: Path to save results, or None to print to console.
    """
    if output_path:
        try:
            _save_json_output(results.model_dump(), output_path)
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")
    else:
        _pretty_print_results(results)


def _pretty_print_agent_results(results: AgentEvaluationResult):
    """Pretty-print agent evaluation results to console."""
    print("\n" + "-" * 80)
    print("Agent Evaluation - Detailed Results:")
    print("-" * 80)
    for i, sr in enumerate(results.sample_results, start=1):
        user_in = sr.get("user_input", "")
        if isinstance(user_in, dict):
            user_in = json.dumps(user_in)
        elif len(str(user_in)) > 80:
            user_in = str(user_in)[:80] + "..."
        print(f"\nSample {i}: {user_in}")
        for metric_key, metric_data in sr.get("metrics", {}).items():
            score = metric_data.get("score")
            passed = metric_data.get("metadata", {}).get("passed")
            err = metric_data.get("metadata", {}).get("error")

            if score is None:
                print(f"  ❌ {metric_key}: N/A")
                if err:
                    print(f"     Error: {err}")
            else:
                status = "PASSED" if passed else "WARNING"
                print(f"  {status} {metric_key}: {score:.4f}")
                if err:
                    print(f"     Warning: {err}")
    if results.summary:
        print("\n" + "-" * 80)
        print("Summary:")
        for k, v in results.summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 80)


def output_agent_results(results: AgentEvaluationResult, output_path: Path | None):
    """Save or print agent evaluation results."""
    if output_path:
        try:
            _save_json_output(results.model_dump(), output_path)
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")
    else:
        _pretty_print_agent_results(results)


def output_workflow_results(results: dict, output_path: Path | None):
    """Save or print workflow CLI evaluation results."""
    if output_path:
        try:
            _save_json_output(results, output_path)
        except Exception as e:
            print(f"Error saving results to {output_path}: {e}")
    else:
        print(json.dumps(results, indent=2, default=str))


def _run_workflow_evaluate(
    dataset: AgentDataset,
    llm_config: OpenAIProviderConfig,
    eval_config: dict,
    workflow_config: dict,
    output_file: Path | None,
):
    """Run workflow evaluation via WorkflowExecutor + WorkflowEvaluation."""
    try:
        from floeval.flotorch.dag import DAG
        from floeval.flotorch.workflow_executor import WorkflowExecutor
    except ImportError as e:
        raise ConfigError(
            f"FloTorch integration required for workflow CLI mode. Import failed: {e}"
        ) from e

    dag_config = workflow_config.get("config") if isinstance(workflow_config, dict) else None
    if not dag_config:
        raise ConfigError(
            "agent_workflow_config.config is required for workflow evaluation."
        )

    # Accept both old builder key `agentName` and runtime key `callable_name`.
    # DAG.from_builder_json expects `callableName`.
    normalized_dag_config = dict(dag_config)
    normalized_nodes: list[dict] = []
    for node in normalized_dag_config.get("nodes", []):
        if not isinstance(node, dict):
            normalized_nodes.append(node)
            continue
        node_copy = dict(node)
        if "callableName" not in node_copy:
            if "agentName" in node_copy:
                node_copy["callableName"] = node_copy["agentName"]
            elif "callable_name" in node_copy:
                node_copy["callableName"] = node_copy["callable_name"]
        normalized_nodes.append(node_copy)
    normalized_dag_config["nodes"] = normalized_nodes

    if not dataset.all_partial:
        raise ConfigError(
            "Workflow CLI evaluation currently expects a partial agent dataset "
            "(samples with user_input/reference_outcome)."
        )

    metrics = list(eval_config.get("metrics") or [])
    if not metrics:
        raise ConfigError(
            "metrics are required for workflow evaluation. "
            "Add 'metrics' to evaluation_config."
        )

    per_sample_results: list[dict] = []
    aggregate_totals: dict[str, float] = defaultdict(float)
    aggregate_counts: dict[str, int] = defaultdict(int)

    for idx, sample in enumerate(dataset.all_partial):
        dag = DAG.from_builder_json(
            {"config": normalized_dag_config, "invocationId": f"cli-wf-{uuid4()}"}
        )
        executor = WorkflowExecutor(
            dag=dag,
            llm_config=llm_config,
            app_name="floeval-workflow-cli",
            user_id=f"cli-user-{idx}",
        )
        execution = run_coroutine_sync(
            lambda s=sample: executor.execute_and_build(
                wf_input=s.user_input,
                reference_outcome=s.reference_outcome,
            )
        )
        wf_result = WorkflowEvaluation(
            execution=execution,
            metrics=metrics,
            llm_config=llm_config,
            metric_params=eval_config.get("metric_params", {}),
        ).run()
        result_payload = wf_result.model_dump()
        per_sample_results.append(
            {
                "sample_index": idx,
                "user_input": sample.user_input,
                "result": result_payload,
            }
        )
        for key, value in wf_result.aggregate_scores.items():
            aggregate_totals[key] += float(value)
            aggregate_counts[key] += 1

    summary = {
        key: round(aggregate_totals[key] / aggregate_counts[key], 4)
        for key in aggregate_totals
        if aggregate_counts[key] > 0
    }
    output_workflow_results(
        {
            "sample_count": len(per_sample_results),
            "sample_results": per_sample_results,
            "summary": summary,
        },
        output_file,
    )


def _run_agent_evaluate(args: argparse.Namespace):
    """Run agent evaluation (Mode 1 or 4)."""
    config_file = args.config
    output_file = Path(args.output) if args.output else None

    _ensure_output_parent(output_file)
    dataset_path = _resolve_dataset_path(args.dataset)

    config_loader = CLIConfigLoader(model_class=CLIEvaluationConfig)
    evaluation_config = config_loader.load(config_file)
    llm_config_dict = evaluation_config.llm_config
    if not llm_config_dict:
        raise ConfigError("Missing 'llm_config' section in the configuration file")

    llm_config = OpenAIProviderConfig(
        base_url=llm_config_dict.get("base_url", "https://api.openai.com/v1"),
        api_key=llm_config_dict["api_key"],
        chat_model=llm_config_dict.get("chat_model", "gpt-3.5-turbo"),
        chat_endpoint=llm_config_dict.get("chat_endpoint", "chat/completions"),
        embedding_model=llm_config_dict.get("embedding_model"),
        embedding_endpoint=llm_config_dict.get("embedding_endpoint"),
        system_prompt=llm_config_dict.get("system_prompt"),
    )

    eval_config = evaluation_config.evaluation_config or {}
    agent_name = eval_config.get("agent_name")
    metrics = eval_config.get("metrics")
    if not metrics:
        raise ConfigError(
            "Missing 'metrics' in evaluation_config for agent evaluation."
        )
    metrics = list(metrics)

    dataset = AgentDataset.from_file(dataset_path)

    workflow_config = getattr(evaluation_config, "agent_workflow_config", None)

    if dataset.is_partial and not agent_name and not workflow_config:
        raise ConfigError(
            "Partial dataset requires either agent_name (single-agent) or "
            "agent_workflow_config (workflow). Add one to your config."
        )

    if workflow_config:
        return _run_workflow_evaluate(
            dataset=dataset,
            llm_config=llm_config,
            eval_config=eval_config,
            workflow_config=workflow_config,
            output_file=output_file,
        )

    agent_runner = None
    if agent_name:
        try:
            from floeval.flotorch import create_flotorch_runner

            agent_runner = create_flotorch_runner(
                agent_name,
                llm_config=llm_config,
            )
        except ImportError as e:
            raise ConfigError(
                f"FloTorch required for Mode 4 (agent_name={agent_name}). "
                f"Import failed: {e}"
            ) from e

    evaluation = AgentEvaluation(
        dataset=dataset,
        metrics=metrics,
        llm_config=llm_config,
        agent_runner=agent_runner,
        metric_params=eval_config.get("metric_params", {}),
    )

    results = evaluation.run()
    output_agent_results(results, output_file)


def parse_args(args: argparse.Namespace):
    if getattr(args, "agent", False):
        return _run_agent_evaluate(args)

    config_file = args.config
    output_file = Path(args.output) if args.output else None

    _ensure_output_parent(output_file)
    dataset_file = _resolve_dataset_path(args.dataset)

    config_loader = CLIConfigLoader(model_class=CLIEvaluationConfig)
    evaluation_config = config_loader.load(config_file)
    llm_config = evaluation_config.llm_config
    if not llm_config:
        raise ConfigError("Missing 'llm_config' section in the configuration file")
    eval_config = evaluation_config.evaluation_config
    if not eval_config:
        raise ConfigError(
            "Missing 'evaluation_config' section in the configuration file"
        )
    llm_config = LLMProviderConfig(
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"],
        chat_model=llm_config["chat_model"],
        embedding_model=llm_config["embedding_model"],
        system_prompt=llm_config.get("system_prompt"),
        chat_endpoint=llm_config.get("chat_endpoint", "chat/completions"),
        embedding_endpoint=llm_config.get("embedding_endpoint", "embeddings"),
    )

    partial_dataset = _is_partial_dataset(Path(dataset_file))
    dataset = DatasetLoader.from_file(dataset_file, partial_dataset=partial_dataset)

    dataset_generator_model = None
    if partial_dataset:
        dg_config = evaluation_config.dataset_generation_config
        if dg_config:
            dataset_generator_model = dg_config.get("generator_model")
        if not dataset_generator_model:
            dataset_generator_model = eval_config.get("dataset_generator_model")
        if not dataset_generator_model:
            raise ConfigError(
                "dataset_generator_model is required for partial datasets (samples without llm_response). "
                "Add 'dataset_generation_config': {'generator_model': 'your-model'} or "
                "'dataset_generator_model' in evaluation_config to your config file."
            )

    evaluation = Evaluation(
        dataset=dataset,
        llm_config=llm_config,
        default_provider=eval_config.get("default_provider"),
        metrics=eval_config["metrics"],
        metric_params=eval_config.get("metric_params", {}),
        dataset_generator_model=dataset_generator_model,
        prompts_file=eval_config.get("prompts_file"),
    )

    results = evaluation.run()

    output_results(results, output_file)
