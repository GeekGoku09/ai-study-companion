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
    format_output_as_json,
    format_output_as_markdown,
    validate_pdf_path,
)


def create_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-study-companion",
        description="AI Study Companion - Production RAG System for Engineering Materials",
    )
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
        help="Optional file path to save the generated output",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (default) or json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat completion model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds (default: 60.0)",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    """Executes the AI study companion RAG pipeline according to parsed arguments."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("Error: OPENAI_API_KEY environment variable is not set.\n")
        return 1

    try:
        validate_pdf_path(args.pdf)
    except (ValueError, FileNotFoundError) as err:
        sys.stderr.write(f"Input Error: {err}\n")
        return 1

    sys.stdout.write(f"--- Processing {args.pdf} ---\n")

    try:
        # 1. Ingestion
        sys.stdout.write("[1/3] Extracting and chunking PDF...\n")
        chunks = ingest_pdf(args.pdf)
        if not chunks:
            sys.stderr.write("Error: No extractable text or paragraphs found in the PDF.\n")
            return 1
        sys.stdout.write(f"      Extracted {len(chunks)} semantic chunks.\n")

        # 2. Retrieval setup
        sys.stdout.write("[2/3] Building vector index...\n")
        retriever = Retriever(timeout=args.timeout)
        retriever.build_index(chunks)

        # 3. Generation based on mode
        sys.stdout.write(f"[3/3] Generating {args.mode} using {args.model}...\n")
        llm_service = LLMService(model=args.model, timeout=args.timeout)

        result_model = None

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics and technical details in this document?",
                k=10,
            )
            result_model = summarizer.generate_summary(
                relevant_chunks or chunks[:10]
            )

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important concepts, definitions, and technical formulas for exam",
                k=10,
            )
            result_model = mcq_gen.generate_mcqs(
                relevant_chunks or chunks[:10],
                num_questions=args.num_mcq,
            )

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics and chapters in this document?",
                k=10,
            )
            summary_result = summarizer.generate_summary(
                relevant_chunks or chunks[:10]
            )
            topics = [t.topic for t in summary_result.topics]
            if not topics:
                topics = ["General Concepts", "Key Formulas and Methods", "Applications"]

            planner = RevisionPlanner(llm_service)
            result_model = planner.generate_plan(topics)

        # Format output
        if args.format == "json":
            formatted_output = format_output_as_json(result_model)
        else:
            formatted_output = format_output_as_markdown(args.mode, result_model)

        sys.stdout.write("\n" + formatted_output + "\n")

        # Optionally save to file
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output + "\n")
            sys.stdout.write(f"\n[Saved output to {args.output}]\n")

        return 0

    except (RuntimeError, ValueError, OSError) as err:
        sys.stderr.write(f"Execution Error: {err}\n")
        return 1


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    exit_code = run_pipeline(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
