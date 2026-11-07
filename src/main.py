import argparse
import os
import sys

from dotenv import load_dotenv

from .ingest import ingest_pdf
from .llm import LLMService
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .retriever import Retriever
from .summarizer import Summarizer
from .utils import (
    format_mcq_markdown,
    format_plan_markdown,
    format_summary_markdown,
    serialize_model_json,
)


def create_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser."""
    parser = argparse.ArgumentParser(description="AI Study Companion - RAG System")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the engineering PDF file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "mcq", "plan"],
        required=True,
        help="Operation mode: summary, mcq, or plan",
    )
    parser.add_argument(
        "--num-mcq",
        type=int,
        default=5,
        help="Number of MCQs to generate (for mcq mode, default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write generated output",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown or json (default: markdown)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model for generation (default: gpt-4o-mini or OPENAI_MODEL)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for API requests (default: 60.0)",
    )
    return parser


def validate_args(args: argparse.Namespace):
    """Validates command-line arguments."""
    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF file not found: {args.pdf}")
    if not os.path.isfile(args.pdf):
        raise ValueError(f"Specified path is not a file: {args.pdf}")
    if args.num_mcq < 1 or args.num_mcq > 50:
        raise ValueError("--num-mcq must be between 1 and 50.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0 seconds.")


def run_pipeline(args: argparse.Namespace) -> str:
    """Executes the RAG pipeline based on parsed arguments and returns formatted string output."""
    validate_args(args)

    print(f"--- Processing {args.pdf} ---")

    # 1. Ingestion
    print("[1/3] Extracting and chunking PDF...")
    chunks = ingest_pdf(args.pdf)
    print(f"      Extracted {len(chunks)} semantic chunks.")

    # 2. Retrieval setup
    print("[2/3] Building vector index...")
    retriever = Retriever(timeout=args.timeout)
    retriever.build_index(chunks)

    # 3. Generation based on mode
    print(f"[3/3] Generating {args.mode} using {args.model}...")
    llm_service = LLMService(model=args.model, timeout=args.timeout)

    if args.mode == "summary":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve(
            "What are the main topics, key concepts, formulas, and technical details in this document?",
            k=10,
        )
        result = summarizer.generate_summary(relevant_chunks)
        output_str = (
            serialize_model_json(result)
            if args.format == "json"
            else format_summary_markdown(result)
        )

    elif args.mode == "mcq":
        mcq_gen = MCQGenerator(llm_service)
        relevant_chunks = retriever.retrieve(
            "important concepts, definitions, and technical formulas for exam",
            k=10,
        )
        result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
        output_str = (
            serialize_model_json(result)
            if args.format == "json"
            else format_mcq_markdown(result)
        )

    elif args.mode == "plan":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve(
            "What are the main topics and chapters in this document?",
            k=10,
        )
        summary_result = summarizer.generate_summary(relevant_chunks)
        topics = [t.topic for t in summary_result.topics]

        planner = RevisionPlanner(llm_service)
        result = planner.generate_plan(topics)
        output_str = (
            serialize_model_json(result)
            if args.format == "json"
            else format_plan_markdown(result)
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return output_str


def main():
    load_dotenv()

    parser = create_parser()
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("Error: OPENAI_API_KEY environment variable not set.\n")
        sys.exit(1)

    try:
        output_str = run_pipeline(args)

        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"\nSuccessfully wrote {args.mode} output to: {args.output}")
        else:
            print("\n" + output_str)

    except (FileNotFoundError, ValueError) as e:
        sys.stderr.write(f"Validation Error: {e}\n")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"An unexpected error occurred: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
