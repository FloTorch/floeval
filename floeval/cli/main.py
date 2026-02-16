import argparse

from floeval.cli import parse_evaluate, parse_generate


def handle_command(args: argparse.Namespace):
    if args.command == "evaluate":
        parse_evaluate.parse_args(args)
    elif args.command == "generate":
        parse_generate.parse_args(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def main():
    parser = argparse.ArgumentParser(description="floeval cli")
    parser.add_argument("--version", action="version", version="floeval 0.1.0")

    sub_parsers = parser.add_subparsers(
        description="Run evaluations with floeval", dest="command"
    )
    # ---- subcommand for running evaluations ----
    evaluate = sub_parsers.add_parser("evaluate", help="Run evaluations with floeval")
    evaluate.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the evaluation configuration file (supported formats: YAML or JSON)",
    )
    evaluate.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to the dataset file (supported formats: JSON, JSONL)",
        required=True,
    )
    evaluate.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save evaluation results (optional)",
        default=None,
    )
    # ----- subcommand for dataset generation from partial dataset -----
    generate = sub_parsers.add_parser(
        "generate", help="Generate complete dataset from partial dataset"
    )
    generate.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the generation configuration file (supported formats: YAML or JSON)",
    )
    generate.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to the partial dataset file",
        required=True,
    )
    generate.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the complete dataset with generated LLM responses (optional)",
        required=True,
    )

    # ------- set default function to handle commands -------
    evaluate.set_defaults(func=handle_command)
    generate.set_defaults(func=handle_command)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    SystemExit(main())
