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

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="AI Study Companion - RAG System")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the engineering PDF file")
    parser.add_argument("--mode", type=str, choices=["summary", "mcq", "plan"], required=True, 
                        help="Operation mode: summary, mcq, or plan")
    parser.add_argument("--num-mcq", type=int, default=5, help="Number of MCQs to generate (for mcq mode)")
    
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print(f"--- Processing {args.pdf} ---")
    
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
        llm_service = LLMService()
        
        if args.mode == "summary":
            summarizer = Summarizer(llm_service)
            # For summary, we might want to use a broader context or specific queries.
            # Here we just use all chunks if they are small enough, or retrieve key ones.
            # For simplicity, let's retrieve top chunks for a 'general summary' query.
            relevant_chunks = retriever.retrieve("What are the main topics and technical details in this document?", k=10)
            result = summarizer.generate_summary(relevant_chunks)
            
            print("\n=== TOPIC-WISE SUMMARY ===")
            for topic in result.topics:
                print(f"\nTopic: {topic.topic}")
                print(f"Summary: {topic.summary}")

        elif args.mode == "mcq":
            mcq_gen = MCQGenerator(llm_service)
            # Retrieve relevant chunks for MCQ generation
            relevant_chunks = retriever.retrieve("important concepts, definitions, and technical formulas for exam", k=10)
            result = mcq_gen.generate_mcqs(relevant_chunks, num_questions=args.num_mcq)
            
            print(f"\n=== GENERATED {len(result.questions)} MCQs ===")
            for i, q in enumerate(result.questions):
                print(f"\nQ{i+1}: {q.question}")
                for idx, opt in enumerate(q.options):
                    print(f"  {chr(65+idx)}) {opt}")
                print(f"Correct Answer: {q.correct_answer}")
                print(f"Explanation: {q.explanation}")

        elif args.mode == "plan":
            # First get topics to plan for
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

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
