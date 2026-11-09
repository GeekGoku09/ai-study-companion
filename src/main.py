import argparse
import os
import sys
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import (
    format_summary_markdown,
    format_mcq_markdown,
    format_plan_markdown,
    serialize_model_to_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and configures the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Study Companion - Intelligent RAG System for Engineering Coursework"
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
        help="Number of MCQs to generate (for mcq mode, must be positive, default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save generated output file",
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
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for LLM API requests (default: 60.0)",
    )
    return parser


def format_output(result, mode: str, output_format: str) -> str:
    """Formats the result object based on requested output format."""
    if output_format == "json":
        return serialize_model_to_json(result)

    if mode == "summary":
        return format_summary_markdown(result)
    elif mode == "mcq":
        return format_mcq_markdown(result)
    elif mode == "plan":
        return format_plan_markdown(result)

    return str(result)


def run_app(args: argparse.Namespace) -> int:
    """Runs the main RAG processing pipeline."""
    if args.num_mcq <= 0:
        print(f"Error: --num-mcq must be a positive integer, got {args.num_mcq}", file=sys.stderr)
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        return 1

    print(f"--- Processing {args.pdf} ---")

    try:
        # 1. Ingestion
        print("[1/3] Extracting and chunking PDF...")
        chunks = ingest_pdf(args.pdf)
        print(f"      Extracted {len(chunks)} semantic chunks.")

        # 2. Retrieval setup
        print("[2/3] Building vector index...")
        retriever = Retriever(timeout=args.timeout)
        retriever.build_index(chunks)

        # 3. Generation based on mode
        print(f"[3/3] Generating {args.mode} (model: {args.model})...")
        llm_service = LLMService(model=args.model, timeout=args.timeout)

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, formulas, principles, and technical details in this document?",
                k=10,
            )
            result = summarizer.generate_summary(relevant_chunks)

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important technical concepts, formulas, definitions, and problem-solving principles for exam",
                k=10,
            )
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, chapters, and core modules in this document?",
                k=10,
            )
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            if not topics:
                # Fallback to general topics if summarization was empty
                topics = ["Core Document Theory", "Applied Problems", "Formula Review"]

            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)

        else:
            print(f"Error: Unsupported mode '{args.mode}'", file=sys.stderr)
            return 1

        formatted_output = format_output(result, args.mode, args.format)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output + "\n")
            print(f"\n[Done] Output written to {args.output}")
        else:
            print("\n" + formatted_output)

        return 0

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1


def main():
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()
    exit_code = run_app(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
