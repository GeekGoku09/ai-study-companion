import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import export_result

logger = logging.getLogger("ai_study_companion")

def configure_logging(verbose: bool = False):
    """Sets up root logging based on verbose flag."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s" if verbose else "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=format_str, force=True)

def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line arguments parser."""
    parser = argparse.ArgumentParser(
        description="AI Study Companion - RAG System for Engineering Materials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pdf",
        type=str,
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
        help="Number of MCQs to generate (used in 'mcq' mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results (supports .md, .markdown, or .json)"
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Optional directory to save the built FAISS index and chunk metadata"
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Optional directory to load a pre-built FAISS index, bypassing PDF ingestion"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging"
    )
    return parser

def main(cli_args=None):
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(cli_args)
    configure_logging(args.verbose)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")
        return 1

    if not args.pdf and not args.load_index:
        logger.error("Either --pdf or --load-index must be provided.")
        return 1

    retriever = Retriever()

    try:
        # 1. Index loading or Ingestion + Index building
        if args.load_index:
            logger.info("Loading vector index from '%s'...", args.load_index)
            retriever.load_index(args.load_index)
            logger.info("Successfully loaded %d chunks from existing index.", len(retriever.chunks))
        else:
            logger.info("--- Processing PDF: %s ---", args.pdf)
            logger.info("[1/3] Extracting and chunking PDF...")
            chunks = ingest_pdf(args.pdf)
            logger.info("Extracted %d semantic chunks.", len(chunks))
            if not chunks:
                logger.error("No valid text chunks could be extracted from '%s'.", args.pdf)
                return 1

            logger.info("[2/3] Building vector index...")
            retriever.build_index(chunks)

            if args.save_index:
                logger.info("Saving index to '%s'...", args.save_index)
                retriever.save_index(args.save_index)
                logger.info("Index successfully saved.")

        # 2. Generation based on mode
        logger.info("[3/3] Generating %s...", args.mode)
        llm_service = LLMService()
        result_obj = None

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, core concepts, formulas, and technical details in this document?",
                k=10
            )
            result = summarizer.generate_summary(relevant_chunks)
            result_obj = result
            
            print("\n=== TOPIC-WISE SUMMARY ===")
            for topic in result.topics:
                print(f"\nTopic: {topic.topic}")
                print(f"Summary: {topic.summary}")

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important concepts, definitions, engineering principles, and technical formulas for exam",
                k=10
            )
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            result_obj = result
            
            print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
            for i, q in enumerate(result.questions):
                print(f"\nQ{i+1}: {q.question}")
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
            
            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)
            result_obj = result
            
            print("\n=== 7-DAY REVISION PLAN ===")
            for day_plan in result.plan:
                print(f"\nDay {day_plan.day}:")
                print(f"  Topics: {', '.join(day_plan.topics)}")
                print(f"  Activities: {', '.join(day_plan.activities)}")
                print(f"  Estimated Time: {day_plan.estimated_time_hours} hours")

        # 3. Export to output file if requested
        if args.output and result_obj is not None:
            saved_path = export_result(result_obj, args.output)
            logger.info("Saved %s output to: %s", args.mode, saved_path)

        return 0

    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=args.verbose)
        return 1

if __name__ == "__main__":
    sys.exit(main())
