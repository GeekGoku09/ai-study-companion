import argparse
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService, LLMServiceError
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .utils import export_to_json, export_to_markdown

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="AI Study Companion - RAG System for Engineering Documents"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the engineering PDF file (required unless --load-index is specified)"
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
        help="Number of MCQs to generate (for mcq mode, default: 5)"
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Directory path to save the FAISS vector index and metadata"
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Directory path to load an existing FAISS vector index and metadata"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path (.json or .md) to save generated results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please set it or add to .env file.", file=sys.stderr)
        sys.exit(1)

    if not args.pdf and not args.load_index:
        print("Error: Either --pdf or --load-index must be provided.", file=sys.stderr)
        sys.exit(1)

    try:
        retriever = Retriever()

        # Step 1 & 2: Ingestion / Index setup
        if args.load_index:
            print(f"[1/2] Loading vector index from {args.load_index}...")
            retriever.load_index(args.load_index)
            print(f"      Loaded {len(retriever.chunks)} chunks from saved index.")
        else:
            print(f"--- Processing {args.pdf} ---")
            print("[1/3] Extracting and chunking PDF...")
            chunks = ingest_pdf(args.pdf)
            print(f"      Extracted {len(chunks)} semantic chunks.")

            print("[2/3] Building vector index...")
            retriever.build_index(chunks)
            print("      Vector index successfully built.")

            if args.save_index:
                print(f"      Saving vector index to {args.save_index}...")
                retriever.save_index(args.save_index)
                print("      Index saved.")

        # Step 3: Generation based on mode
        llm_service = LLMService()
        result_model = None
        markdown_output = ""

        if args.mode == "summary":
            print("[3/3] Generating topic-wise summary...")
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, core technical concepts, and formulas in this document?",
                k=10
            )
            result = summarizer.generate_summary(relevant_chunks)
            result_model = result

            print("\n=== TOPIC-WISE SUMMARY ===")
            md_lines = ["# Topic-Wise Summary\n"]
            for topic in result.topics:
                print(f"\nTopic: {topic.topic}")
                print(f"Summary: {topic.summary}")
                md_lines.append(f"## {topic.topic}\n\n{topic.summary}\n")
            markdown_output = "\n".join(md_lines)

        elif args.mode == "mcq":
            print(f"[3/3] Generating {args.num_mcq} MCQs...")
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve(
                "important technical definitions, formulas, rules, and exam-style concepts",
                k=10
            )
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            result_model = result

            print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
            md_lines = [f"# Exam-Style MCQs ({len(result.questions)} Questions)\n"]
            for i, q in enumerate(result.questions):
                print(f"\nQ{i+1}: {q.question}")
                md_lines.append(f"### Q{i+1}: {q.question}")
                for idx, opt in enumerate(q.options):
                    letter = chr(65 + idx)
                    print(f"  {letter}) {opt}")
                    md_lines.append(f"- **{letter})** {opt}")
                print(f"Correct Answer: {q.correct_answer}")
                print(f"Explanation: {q.explanation}")
                md_lines.append(f"\n**Correct Answer:** {q.correct_answer}")
                md_lines.append(f"**Explanation:** {q.explanation}\n")
            markdown_output = "\n".join(md_lines)

        elif args.mode == "plan":
            print("[3/3] Generating 7-day revision plan...")
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve(
                "What are the main topics, chapters, and units in this engineering document?",
                k=10
            )
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            
            planner = RevisionPlanner(llm_service)
            result = planner.generate_plan(topics)
            result_model = result

            print("\n=== 7-DAY REVISION PLAN ===")
            md_lines = ["# 7-Day Revision Plan\n"]
            for day_plan in result.plan:
                topics_str = ", ".join(day_plan.topics)
                activities_str = ", ".join(day_plan.activities)
                print(f"\nDay {day_plan.day}:")
                print(f"  Topics: {topics_str}")
                print(f"  Activities: {activities_str}")
                print(f"  Estimated Time: {day_plan.estimated_time_hours} hours")

                md_lines.append(f"## Day {day_plan.day}")
                md_lines.append(f"- **Topics:** {topics_str}")
                md_lines.append(f"- **Activities:** {activities_str}")
                md_lines.append(f"- **Estimated Time:** {day_plan.estimated_time_hours} hours\n")
            markdown_output = "\n".join(md_lines)

        # Output persistence if requested
        if args.output and result_model is not None:
            out_p = Path(args.output)
            if out_p.suffix.lower() == ".json":
                export_to_json(result_model, out_p)
                print(f"\n[Export] Results saved to JSON at: {args.output}")
            elif out_p.suffix.lower() in [".md", ".markdown", ".txt"]:
                export_to_markdown(markdown_output, out_p)
                print(f"\n[Export] Results saved to Markdown at: {args.output}")
            else:
                # Default to JSON
                export_to_json(result_model, out_p)
                print(f"\n[Export] Results saved to: {args.output}")

    except LLMServiceError as e:
        print(f"\nLLM Generation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
