import argparse
import os
import sys
import json
from typing import Optional
from dotenv import load_dotenv

from .ingest import ingest_pdf
from .retriever import Retriever
from .llm import LLMService
from .summarizer import Summarizer
from .mcq_generator import MCQGenerator
from .planner import RevisionPlanner

def format_output_markdown(mode: str, result) -> str:
    """Formats the structured Pydantic result into clean Markdown."""
    lines = []
    if mode == "summary":
        lines.append("# Topic-Wise Summary\n")
        for topic in result.topics:
            lines.append(f"## {topic.topic}\n")
            lines.append(f"{topic.summary}\n")
    elif mode == "mcq":
        lines.append(f"# Practice MCQs ({len(result.questions)} Questions)\n")
        for i, q in enumerate(result.questions):
            lines.append(f"### Q{i+1}: {q.question}\n")
            for idx, opt in enumerate(q.options):
                lines.append(f"- **{chr(65+idx)}**) {opt}")
            lines.append(f"\n**Correct Answer:** {q.correct_answer}\n")
            lines.append(f"**Explanation:** {q.explanation}\n")
    elif mode == "plan":
        lines.append("# 7-Day Revision Plan\n")
        for day_plan in result.plan:
            lines.append(f"## Day {day_plan.day}\n")
            lines.append(f"- **Topics:** {', '.join(day_plan.topics)}")
            lines.append(f"- **Activities:** {', '.join(day_plan.activities)}")
            lines.append(f"- **Estimated Time:** {day_plan.estimated_time_hours} hours\n")
    return "\n".join(lines)

def save_output(result, output_path: str, mode: str):
    """Saves result to file as JSON or Markdown depending on file extension."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
    else:
        md_content = format_output_markdown(mode, result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    print(f"\n[+] Output saved successfully to: {output_path}")

def run_pipeline(
    pdf_path: Optional[str],
    mode: str,
    num_mcq: int = 5,
    save_index_path: Optional[str] = None,
    load_index_path: Optional[str] = None,
    output_path: Optional[str] = None,
    model: str = "gpt-4o-mini"
):
    """Main orchestration pipeline for AI Study Companion."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    retriever = Retriever()

    # 1. Indexing / Ingestion phase
    if load_index_path:
        print(f"[1/3] Loading existing index from {load_index_path}...")
        retriever.load_index(load_index_path)
        print(f"      Loaded {len(retriever.chunks)} chunks from index.")
    else:
        if not pdf_path:
            raise ValueError("Must provide --pdf when not loading from an existing index.")
        print(f"[1/3] Ingesting PDF: {pdf_path}...")
        chunks = ingest_pdf(pdf_path)
        if not chunks:
            raise ValueError(f"No readable text chunks could be extracted from {pdf_path}.")
        print(f"      Extracted {len(chunks)} semantic chunks.")

        print("[2/3] Building vector index...")
        retriever.build_index(chunks)
        if save_index_path:
            print(f"      Saving vector index to {save_index_path}...")
            retriever.save_index(save_index_path)

    # 2. Generation phase
    print(f"[3/3] Generating {mode} using {model}...")
    llm_service = LLMService(model=model)
    result = None

    if mode == "summary":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics, theorems, and technical concepts?", k=10)
        result = summarizer.generate_summary(relevant_chunks)
        
        print("\n=== TOPIC-WISE SUMMARY ===")
        for topic in result.topics:
            print(f"\nTopic: {topic.topic}")
            print(f"Summary: {topic.summary}")

    elif mode == "mcq":
        mcq_gen = MCQGenerator(llm_service)
        relevant_chunks = retriever.retrieve("important concepts, definitions, and technical formulas for exam", k=10)
        result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=num_mcq)

        print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
        for i, q in enumerate(result.questions):
            print(f"\nQ{i+1}: {q.question}")
            for idx, opt in enumerate(q.options):
                print(f"  {chr(65+idx)}) {opt}")
            print(f"Correct Answer: {q.correct_answer}")
            print(f"Explanation: {q.explanation}")

    elif mode == "plan":
        summarizer = Summarizer(llm_service)
        relevant_chunks = retriever.retrieve("What are the main topics and chapters in this document?", k=10)
        summary_result = summarizer.generate_summary(relevant_chunks)
        topics = [t.topic for t in summary_result.topics]

        planner = RevisionPlanner(llm_service)
        result = planner.generate_plan(topics)

        print("\n=== 7-DAY REVISION PLAN ===")
        for day_plan in result.plan:
            print(f"\nDay {day_plan.day}:")
            print(f"  Topics: {', '.join(day_plan.topics)}")
            print(f"  Activities: {', '.join(day_plan.activities)}")
            print(f"  Estimated Time: {day_plan.estimated_time_hours} hours")

    if output_path and result:
        save_output(result, output_path, mode)

    return result

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="AI Study Companion - RAG System")
    parser.add_argument("--pdf", type=str, default=None, help="Path to the engineering PDF file")
    parser.add_argument("--mode", type=str, choices=["summary", "mcq", "plan"], required=True,
                        help="Operation mode: summary, mcq, or plan")
    parser.add_argument("--num-mcq", type=int, default=5, help="Number of MCQs to generate (for mcq mode)")
    parser.add_argument("--save-index", type=str, default=None, help="Directory to save the built FAISS index")
    parser.add_argument("--load-index", type=str, default=None, help="Directory to load a pre-built FAISS index from")
    parser.add_argument("--output", type=str, default=None, help="Path to write output (e.g., summary.md or notes.json)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")

    args = parser.parse_args()

    try:
        run_pipeline(
            pdf_path=args.pdf,
            mode=args.mode,
            num_mcq=args.num_mcq,
            save_index_path=args.save_index,
            load_index_path=args.load_index,
            output_path=args.output,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
