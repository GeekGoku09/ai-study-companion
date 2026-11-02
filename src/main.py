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
from .utils import export_to_markdown, export_to_json, save_output_to_file

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("ai_study_companion")

def setup_cli() -> argparse.ArgumentParser:
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
        help="Number of MCQs to generate (for mcq mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save the generated output (e.g., output.md, output.json)"
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Directory to save the FAISS vector index and metadata for reuse"
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Directory to load a pre-built FAISS vector index and metadata from"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging"
    )
    return parser

def main():
    load_dotenv()
    parser = setup_cli()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")
        sys.exit(1)

    if not args.pdf and not args.load_index:
        logger.error("Either --pdf or --load-index must be provided.")
        sys.exit(1)

    if args.num_mcq <= 0:
        logger.error("--num-mcq must be a positive integer.")
        sys.exit(1)

    try:
        retriever = Retriever()

        if args.load_index:
            logger.info(f"Loading pre-computed index from {args.load_index}...")
            retriever.load_index(args.load_index)
        else:
            logger.info(f"Extracting and chunking PDF: {args.pdf}...")
            chunks = ingest_pdf(args.pdf)
            logger.info(f"Extracted {len(chunks)} semantic chunks.")

            logger.info("Building vector index...")
            retriever.build_index(chunks)

            if args.save_index:
                logger.info(f"Saving vector index to {args.save_index}...")
                retriever.save_index(args.save_index)

        logger.info(f"Generating {args.mode}...")
        llm_service = LLMService()
        result_object = None

        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics and technical details in this document?", k=10)
            result_object = summarizer.generate_summary(relevant_chunks)
            print("\n" + export_to_markdown(result_object))

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            relevant_chunks = retriever.retrieve("important concepts, definitions, and technical formulas for exam", k=10)
            result_object = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            print("\n" + export_to_markdown(result_object))

        elif args.mode == "plan":
            summarizer = Summarizer(llm_service)
            relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
            summary_result = summarizer.generate_summary(relevant_chunks)
            topics = [t.topic for t in summary_result.topics]
            
            planner = RevisionPlanner(llm_service)
            result_object = planner.generate_plan(topics)
            print("\n" + export_to_markdown(result_object))

        if args.output and result_object is not None:
            if args.output.endswith(".json"):
                content = export_to_json(result_object)
            else:
                content = export_to_markdown(result_object)
            save_output_to_file(content, args.output)
            logger.info(f"Output successfully saved to {args.output}")

    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
