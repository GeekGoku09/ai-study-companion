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
from .utils import write_output_file

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Study Companion - Production-grade RAG System for Technical Notes"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the engineering PDF file (required unless --load-index is used)"
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
        help="Number of MCQs to generate (default: 5)"
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Path to save the computed FAISS vector index and chunk metadata"
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Path to load a previously saved FAISS vector index and chunk metadata"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write generated output (Markdown or JSON based on --format)"
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
        default=None,
        help="LLM model name to use (default: env LLM_MODEL or gpt-4o-mini)"
    )
    return parser

def run_app(args: argparse.Namespace) -> int:
    """Core execution logic for the AI Study Companion application."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("Error: OPENAI_API_KEY environment variable not set.\n")
        return 1

    if not args.pdf and not args.load_index:
        sys.stderr.write("Error: Either --pdf or --load-index must be specified.\n")
        return 1

    retriever = Retriever()

    # Step 1: Ingestion or Load Index
    if args.load_index:
        print(f"[1/3] Loading vector index from {args.load_index}...")
        try:
            retriever.load_index(args.load_index)
            print(f"      Loaded {len(retriever.chunks)} chunks from index.")
        except Exception as e:
            sys.stderr.write(f"Error loading index from {args.load_index}: {e}\n")
            return 1
    else:
        print(f"[1/3] Extracting and chunking PDF: {args.pdf}...")
        try:
            chunks = ingest_pdf(args.pdf)
            if not chunks:
                sys.stderr.write(f"Warning: No valid text chunks could be extracted from {args.pdf}.\n")
                return 1
            print(f"      Extracted {len(chunks)} semantic chunks.")
        except Exception as e:
            sys.stderr.write(f"Error processing PDF {args.pdf}: {e}\n")
            return 1

        print("[2/3] Building vector index...")
        try:
            retriever.build_index(chunks)
        except Exception as e:
            sys.stderr.write(f"Error building vector index: {e}\n")
            return 1

        if args.save_index:
            print(f"      Saving vector index to {args.save_index}...")
            try:
                retriever.save_index(args.save_index)
                print(f"      Index successfully saved.")
            except Exception as e:
                sys.stderr.write(f"Error saving index to {args.save_index}: {e}\n")
                return 1

    # Step 2: Generation based on mode
    print(f"[3/3] Generating {args.mode}...")
    try:
        llm_service = LLMService(model=args.model) if args.model else LLMService()

        formatted_output = ""

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, key concepts, and technical details in this document?",
                k=10
            )
            result = summarizer.generate_summary(relevant_chunks)
            if args.format == "json":
                formatted_output = result.to_json()
            else:
                formatted_output = result.to_markdown()

            print("\n=== TOPIC-WISE SUMMARY ===")
            for topic in result.topics:
                print(f"\nTopic: {topic.topic}")
                print(f"Summary: {topic.summary}")

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important concepts, definitions, formulas, and principles for exam questions",
                k=10
            )
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            if args.format == "json":
                formatted_output = result.to_json()
            else:
                formatted_output = result.to_markdown()

            print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
            for i, q in enumerate(result.questions, 1):
                print(f"\nQ{i}: {q.question}")
                for idx, opt in enumerate(q.options):
                    print(f"  {chr(65+idx)}) {opt}")
                print(f"Correct Answer: {q.correct_answer}")
                print(f"Explanation: {q.explanation}")

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, chapters, and sections in this document?",
                k=10
            )
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            if not topics:
                topics = ["General Engineering Concepts", "Core Theories and Problem Solving"]

            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)
            if args.format == "json":
                formatted_output = result.to_json()
            else:
                formatted_output = result.to_markdown()

            print("\n=== 7-DAY REVISION PLAN ===")
            for day_plan in result.plan:
                print(f"\nDay {day_plan.day}:")
                print(f"  Topics: {', '.join(day_plan.topics)}")
                print(f"  Activities: {', '.join(day_plan.activities)}")
                print(f"  Estimated Time: {day_plan.estimated_time_hours} hours")

        if args.output:
            write_output_file(args.output, formatted_output)
            print(f"\nOutput successfully written to: {args.output}")

        return 0

    except Exception as e:
        sys.stderr.write(f"An error occurred during generation: {e}\n")
        return 1

def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    exit_code = run_app(args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
