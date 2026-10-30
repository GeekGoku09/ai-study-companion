import argparse
import os
import sys
import json
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import setup_logger
from .exceptions import StudyCompanionError

logger = setup_logger("study_companion_cli")

def format_summary_markdown(result) -> str:
    lines = ["# Topic-Wise Summary\n"]
    for t in result.topics:
        lines.append(f"## {t.topic}\n")
        lines.append(f"{t.summary}\n")
    return "\n".join(lines)

def format_mcq_markdown(result) -> str:
    lines = [f"# Exam Practice MCQs ({len(result.questions)} questions)\n"]
    for i, q in enumerate(result.questions):
        lines.append(f"### Q{i+1}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            lines.append(f"- **{chr(65+idx)})** {opt}")
        lines.append(f"\n> **Correct Answer:** {q.correct_answer}\n")
        lines.append(f"> **Explanation:** {q.explanation}\n")
    return "\n".join(lines)

def format_plan_markdown(result) -> str:
    lines = ["# 7-Day Study Revision Plan\n"]
    for day in result.plan:
        lines.append(f"## Day {day.day} ({day.estimated_time_hours} hours)\n")
        lines.append(f"- **Topics:** {', '.join(day.topics)}")
        lines.append(f"- **Activities:** {', '.join(day.activities)}\n")
    return "\n".join(lines)

def write_output_file(output_path: str, data: dict, markdown_content: str):
    """Saves output as either JSON or Markdown depending on file extension."""
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
    logger.info(f"Output saved successfully to: {output_path}")

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Study Companion - RAG System for Engineering PDFs"
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
        help="Number of MCQs to generate in mcq mode (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save output file (.json, .md, or .txt)"
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Optional directory path to persist the built vector index and chunks"
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Optional directory path to load a pre-built vector index, skipping PDF parsing"
    )
    return parser

def main():
    load_dotenv()
    parser = build_cli_parser()
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set. Please set it in .env or export it.")
        sys.exit(1)

    if not args.pdf and not args.load_index:
        logger.error("Either --pdf or --load-index must be specified.")
        parser.print_usage()
        sys.exit(1)

    try:
        retriever = Retriever()

        # 1. Setup Retrieval Index (either load or ingest PDF)
        if args.load_index:
            logger.info(f"[1/3] Loading cached index from {args.load_index}...")
            retriever.load_index(args.load_index)
            print(f"Loaded {len(retriever.chunks)} indexed chunks from cache.")
        else:
            logger.info(f"[1/3] Ingesting and chunking {args.pdf}...")
            chunks = ingest_pdf(args.pdf)
            print(f"Extracted {len(chunks)} semantic chunks.")
            if not chunks:
                logger.error("No valid text chunks could be extracted from the PDF.")
                sys.exit(1)

            logger.info("[2/3] Building vector index...")
            retriever.build_index(chunks)
            if args.save_index:
                retriever.save_index(args.save_index)
                print(f"Index successfully saved to {args.save_index}")

        # 2. Execute selected generation mode
        logger.info(f"[3/3] Generating {args.mode}...")
        llm_service = LLMService()

        output_data = {}
        markdown_output = ""

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics and technical details in this document?", k=10)
            result = summarizer.generate_summary(relevant_chunks)

            output_data = result.model_dump()
            markdown_output = format_summary_markdown(result)

            print("\n=== TOPIC-WISE SUMMARY ===")
            for topic in result.topics:
                print(f"\nTopic: {topic.topic}")
                print(f"Summary: {topic.summary}")

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve("important concepts, definitions, and technical formulas for exam", k=10)
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)

            output_data = result.model_dump()
            markdown_output = format_mcq_markdown(result)

            print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
            for i, q in enumerate(result.questions):
                print(f"\nQ{i+1}: {q.question}")
                for idx, opt in enumerate(q.options):
                    print(f"  {chr(65+idx)}) {opt}")
                print(f"Correct Answer: {q.correct_answer}")
                print(f"Explanation: {q.explanation}")

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]

            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)

            output_data = result.model_dump()
            markdown_output = format_plan_markdown(result)

            print("\n=== 7-DAY REVISION PLAN ===")
            for day_plan in result.plan:
                print(f"\nDay {day_plan.day}:")
                print(f"  Topics: {', '.join(day_plan.topics)}")
                print(f"  Activities: {', '.join(day_plan.activities)}")
                print(f"  Estimated Time: {day_plan.estimated_time_hours} hours")

        # 3. Export to file if requested
        if args.output:
            write_output_file(args.output, output_data, markdown_output)
            print(f"\nResult exported to: {args.output}")

    except StudyCompanionError as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
