import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .llm import LLMService
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .retriever import Retriever
from .summarizer import Summarizer
from .utils import export_to_json, export_to_markdown, write_output_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("study_companion")


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="AI Study Companion - Production RAG System for Engineering Materials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Number of MCQs to generate (for mcq mode)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown or json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save the generated output",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI chat completion model",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for LLM API calls",
    )
    return parser.parse_args(args)


def run_pipeline(args) -> str:
    """Runs the RAG study companion pipeline and returns the formatted output string."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running.")

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"Specified PDF file does not exist: {args.pdf}")

    logger.info("Processing document: %s", args.pdf)

    # 1. Ingestion
    logger.info("[1/3] Extracting and chunking PDF...")
    chunks = ingest_pdf(args.pdf)
    if not chunks:
        raise ValueError(
            f"No readable text chunks could be extracted from '{args.pdf}'. "
            "Ensure the PDF contains selectable text."
        )
    logger.info("Extracted %d semantic chunks.", len(chunks))

    # 2. Vector indexing
    logger.info("[2/3] Building vector index...")
    retriever = Retriever(api_key=api_key)
    retriever.build_index(chunks)

    # 3. Generation
    logger.info("[3/3] Generating %s using model %s...", args.mode, args.model)
    llm_service = LLMService(api_key=api_key, model=args.model, timeout=args.timeout)

    result_data = None
    if args.mode == "summary":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics and technical details in this document?", k=10)
        result_data = summarizer.generate_summary(relevant_chunks)

    elif args.mode == "mcq":
        mcq_gen = MCQGenerator(llm_service)
        relevant_chunks = retriever.retrieve("important concepts, definitions, formulas, and principles for exam", k=10)
        result_data = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)

    elif args.mode == "plan":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
        summary_result = summarizer.generate_summary(relevant_chunks)
        topics = [t.topic for t in summary_result.topics]
        planner = RevisionPlanner(llm_service)
        result_data = planner.generate_plan(topics)

    # Format output
    if args.format == "json":
        formatted_output = export_to_json(result_data)
    else:
        formatted_output = export_to_markdown(result_data)

    # Save to file if requested
    if args.output:
        write_output_file(formatted_output, args.output)
        logger.info("Output successfully written to: %s", args.output)

    return formatted_output


def main():
    load_dotenv()
    args = parse_arguments()

    try:
        output = run_pipeline(args)
        print("\n" + output)
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
