import argparse
import json
import os
import sys

from dotenv import load_dotenv

from .ingest import ingest_pdf
from .llm import LLMService
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner
from .retriever import Retriever
from .schemas import MCQList, RevisionPlan, SummaryResponse
from .summarizer import Summarizer
from .utils import check_file_exists


def format_summary_markdown(result: SummaryResponse) -> str:
    """Formats SummaryResponse as Markdown."""
    lines = ["# Topic-Wise Summary\n"]
    if not result.topics:
        lines.append("*No topics found.*")
    for topic in result.topics:
        lines.append(f"## {topic.topic}\n")
        lines.append(f"{topic.summary}\n")
    return "\n".join(lines).strip()


def format_mcqs_markdown(result: MCQList) -> str:
    """Formats MCQList as Markdown."""
    lines = [f"# Exam-Style MCQs ({len(result.questions)} Questions)\n"]
    if not result.questions:
        lines.append("*No questions generated.*")
    for i, q in enumerate(result.questions, start=1):
        lines.append(f"### Q{i}: {q.question}\n")
        for idx, opt in enumerate(q.options):
            lines.append(f"- **{chr(65 + idx)})** {opt}")
        lines.append(f"\n**Correct Answer:** {q.correct_answer}")
        lines.append(f"**Explanation:** {q.explanation}\n")
    return "\n".join(lines).strip()


def format_plan_markdown(result: RevisionPlan) -> str:
    """Formats RevisionPlan as Markdown."""
    lines = ["# 7-Day Revision Plan\n"]
    if not result.plan:
        lines.append("*No revision plan generated.*")
    for day_plan in result.plan:
        lines.append(f"## Day {day_plan.day}")
        lines.append(f"**Estimated Time:** {day_plan.estimated_time_hours} hours\n")
        lines.append("**Topics:**")
        for t in day_plan.topics:
            lines.append(f"- {t}")
        lines.append("\n**Activities:**")
        for a in day_plan.activities:
            lines.append(f"- {a}")
        lines.append("")
    return "\n".join(lines).strip()


def format_output(result, mode: str, output_format: str) -> str:
    """Formats the companion result into either Markdown or JSON."""
    if output_format == "json":
        return json.dumps(result.model_dump(), indent=2)

    if mode == "summary":
        return format_summary_markdown(result)
    elif mode == "mcq":
        return format_mcqs_markdown(result)
    elif mode == "plan":
        return format_plan_markdown(result)
    return str(result)


def run_pipeline(
    pdf_path: str,
    mode: str,
    num_mcq: int = 5,
    model: str = "gpt-4o-mini",
    timeout: float | None = None,
    output_format: str = "markdown",
    output_path: str | None = None,
) -> str:
    """Executes the study companion RAG pipeline and returns the formatted output string."""
    if not check_file_exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"--- Processing {pdf_path} ---")

    # 1. Ingestion
    print("[1/3] Extracting and chunking PDF...")
    chunks = ingest_pdf(pdf_path)
    print(f"      Extracted {len(chunks)} semantic chunks.")

    # 2. Retrieval setup
    print("[2/3] Building vector index...")
    retriever = Retriever(timeout=timeout)
    retriever.build_index(chunks)

    # 3. Generation based on mode
    print(f"[3/3] Generating {mode} using {model}...")
    llm_service = LLMService(model=model, timeout=timeout)

    if mode == "summary":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics and technical details in this document?", k=10)
        result = summarizer.generate_summary(relevant_chunks)

    elif mode == "mcq":
        mcq_gen = MCQGenerator(llm_service)
        relevant_chunks = retriever.retrieve("important concepts, definitions, and technical formulas for exam", k=10)
        result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=num_mcq)

    elif mode == "plan":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
        summary_result = summarizer.generate_summary(relevant_chunks)
        topics = [t.topic for t in summary_result.topics]

        planner = RevisionPlanner(llm_service)
        result = planner.generate_plan(topics)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    formatted_text = format_output(result, mode, output_format)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        print(f"\nSaved output to {output_path}")

    return formatted_text


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="AI Study Companion - RAG System")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the engineering PDF file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "mcq", "plan"],
        required=True,
        help="Operation mode: summary, mcq, or plan",
    )
    parser.add_argument("--num-mcq", type=int, default=5, help="Number of MCQs to generate (for mcq mode)")
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (default) or json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination file path to save the generated output",
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
        default=None,
        help="Timeout in seconds for API network requests",
    )
    return parser.parse_args(args)


def main():
    load_dotenv()
    args = parse_arguments()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        output = run_pipeline(
            pdf_path=args.pdf,
            mode=args.mode,
            num_mcq=args.num_mcq,
            model=args.model,
            timeout=args.timeout,
            output_format=args.format,
            output_path=args.output,
        )
        print(f"\n{output}")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
