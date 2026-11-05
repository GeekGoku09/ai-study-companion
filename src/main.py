import argparse
import os
import sys
import logging
from dotenv import load_dotenv

from .ingest import ingest_pdf, IngestionError
from .retriever import Retriever
from .llm import LLMService
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import export_to_markdown, export_to_json

logger = logging.getLogger(__name__)

def setup_logging(level_name: str = "INFO"):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level
    )

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="AI Study Companion - Production-Ready RAG System for Engineering Students"
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to the engineering PDF file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "mcq", "plan"],
        required=True,
        help="Operation mode: summary, mcq, or plan"
    )
    parser.add_argument("--num-mcq", type=int, default=5, help="Number of MCQs to generate (for mcq mode, 1-50)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output filepath to save results (.md or .json)")
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Export format when saving with --output (default: markdown)"
    )
    parser.add_argument("--model", type=str, default=None, help="OpenAI chat model (default: gpt-4o-mini)")
    parser.add_argument("--timeout", type=float, default=60.0, help="LLM request timeout in seconds (default: 60)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")
        sys.exit(1)

    if args.num_mcq < 1 or args.num_mcq > 50:
        logger.error(f"Invalid --num-mcq value: {args.num_mcq}. Must be between 1 and 50.")
        sys.exit(1)

    if not os.path.exists(args.pdf):
        logger.error(f"File not found: {args.pdf}")
        sys.exit(1)

    print(f"\n--- AI Study Companion: Processing {os.path.basename(args.pdf)} ---")
    
    try:
        # 1. Ingestion
        print("[1/3] Extracting and chunking PDF...")
        chunks = ingest_pdf(args.pdf)
        print(f"      Extracted {len(chunks)} semantic chunks.")

        # 2. Retrieval setup
        print("[2/3] Building vector index...")
        retriever = Retriever()
        retriever.build_index(chunks)

        # 3. Generation based on mode
        print(f"[3/3] Generating {args.mode}...")
        llm_service = LLMService(model=args.model, timeout=args.timeout)
        result = None

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics, theorems, and technical formulas in this document?", k=10)
            result = summarizer.generate_summary(relevant_chunks)
            
            print("\n" + "=" * 30 + " TOPIC-WISE SUMMARY " + "=" * 30)
            for topic in result.topics:
                print(f"\n### {topic.topic}")
                print(f"{topic.summary}")

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve("important concepts, definitions, formulas, and derivations for examination", k=10)
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            
            print("\n" + "=" * 25 + f" GENERATED {len(result.questions)} MCQs " + "=" * 25)
            for i, q in enumerate(result.questions, start=1):
                print(f"\nQ{i}: {q.question}")
                for idx, opt in enumerate(q.options):
                    print(f"  {chr(65 + idx)}) {opt}")
                print(f"  [Correct Answer]: {q.correct_answer}")
                print(f"  [Explanation]: {q.explanation}")

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            
            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)
            
            print("\n" + "=" * 25 + " 7-DAY REVISION PLAN " + "=" * 25)
            for day_plan in result.plan:
                print(f"\nDay {day_plan.day} ({day_plan.estimated_time_hours} hrs):")
                print(f"  Topics: {', '.join(day_plan.topics)}")
                print("  Activities:")
                for act in day_plan.activities:
                    print(f"    - {act}")

        # Optional file export
        if args.output and result is not None:
            if args.format == "json" or args.output.endswith(".json"):
                export_to_json(result, args.output)
            else:
                export_to_markdown(result, args.output)
            print(f"\n[Export] Results saved successfully to {args.output}")

    except IngestionError as e:
        logger.error(f"Document ingestion failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=(args.log_level == "DEBUG"))
        sys.exit(1)

if __name__ == "__main__":
    main()
