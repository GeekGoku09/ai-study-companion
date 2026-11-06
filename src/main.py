import argparse
import os
import sys
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService, DEFAULT_MODEL
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import format_result, save_to_file

def build_parser() -> argparse.ArgumentParser:
    """Constructs and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Study Companion - RAG System for Engineering Materials"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the engineering PDF file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "mcq", "plan"],
        required=True,
        help="Operation mode: summary, mcq, or plan"
    )
    parser.add_argument(
        "--num-mcq",
        type=int,
        default=5,
        help="Number of MCQs to generate (for mcq mode, must be >= 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output file (saves markdown or json)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (default) or json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for API calls"
    )
    return parser

def run(args: argparse.Namespace) -> int:
    """Executes the pipeline with parsed CLI arguments."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("Error: OPENAI_API_KEY environment variable not set.\n")
        return 1

    if not os.path.exists(args.pdf):
        sys.stderr.write(f"Error: PDF file not found at {args.pdf}\n")
        return 1

    if args.mode == "mcq" and args.num_mcq <= 0:
        sys.stderr.write("Error: --num-mcq must be greater than 0.\n")
        return 1

    print(f"--- Processing {args.pdf} ---")
    
    try:
        # 1. Ingestion
        print("[1/3] Extracting and chunking PDF...")
        chunks = ingest_pdf(args.pdf)
        if not chunks:
            sys.stderr.write("Error: No readable text extracted from the provided PDF.\n")
            return 1
        print(f"      Extracted {len(chunks)} semantic chunks.")

        # 2. Retrieval setup
        print("[2/3] Building vector index...")
        retriever = Retriever(timeout=args.timeout)
        retriever.build_index(chunks)

        # 3. Generation based on mode
        print(f"[3/3] Generating {args.mode}...")
        llm_service = LLMService(model=args.model, timeout=args.timeout)
        
        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics and technical details in this document?",
                k=10
            )
            result = summarizer.generate_summary(relevant_chunks)

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important concepts, definitions, and technical formulas for exam",
                k=10
            )
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics and chapters in this document?",
                k=10
            )
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)

        # Output formatting
        formatted_output = format_result(result, output_format=args.format)
        print("\n" + formatted_output)

        if args.output:
            save_to_file(args.output, formatted_output)
            print(f"\n[+] Output successfully saved to: {args.output}")

        return 0

    except Exception as e:
        sys.stderr.write(f"An error occurred: {e}\n")
        return 1

def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))

if __name__ == "__main__":
    main()
