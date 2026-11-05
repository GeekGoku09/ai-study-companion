import unittest
from pydantic import ValidationError
from src.schemas import DocumentChunk, MCQ, MCQList, TopicSummary, SummaryResponse, RevisionDay, RevisionPlan

class TestSchemas(unittest.TestCase):
    def test_document_chunk_validation(self):
        chunk = DocumentChunk(content="Valid content", metadata={"page": 1})
        self.assertEqual(chunk.content, "Valid content")
        self.assertEqual(chunk.metadata["page"], 1)

        # Empty content should fail
        with self.assertRaises(ValidationError):
            DocumentChunk(content="")

    def test_mcq_matching_options(self):
        # Case 1: direct option match
        mcq = MCQ(
            question="What is Ohm's law formula?",
            options=["V = I / R", "V = I * R", "P = V * I", "I = V^2 / R"],
            correct_answer="V = I * R",
            explanation="Ohm's law defines voltage as current times resistance."
        )
        self.assertEqual(mcq.correct_answer, "V = I * R")

        # Case 2: letter identifier 'B'
        mcq_b = MCQ(
            question="What is Ohm's law formula?",
            options=["V = I / R", "V = I * R", "P = V * I", "I = V^2 / R"],
            correct_answer="B",
            explanation="Ohm's law defines voltage as current times resistance."
        )
        self.assertEqual(mcq_b.correct_answer, "V = I * R")

        # Case 3: prefixed option "B) V = I * R"
        mcq_prefix = MCQ(
            question="What is Ohm's law formula?",
            options=["V = I / R", "V = I * R", "P = V * I", "I = V^2 / R"],
            correct_answer="B) V = I * R",
            explanation="Ohm's law defines voltage as current times resistance."
        )
        self.assertEqual(mcq_prefix.correct_answer, "V = I * R")

        # Invalid options count
        with self.assertRaises(ValidationError):
            MCQ(
                question="Invalid options?",
                options=["Only one", "Only two"],
                correct_answer="Only one",
                explanation="Fails validation."
            )

    def test_revision_plan_validation(self):
        day = RevisionDay(
            day=1,
            topics=["Kirchhoff Laws"],
            activities=["Derive KVL/KCL equations"],
            estimated_time_hours=2.5
        )
        plan = RevisionPlan(plan=[day])
        self.assertEqual(len(plan.plan), 1)
        self.assertEqual(plan.plan[0].day, 1)

        # Invalid day number (less than 1)
        with self.assertRaises(ValidationError):
            RevisionDay(
                day=0,
                topics=["Invalid"],
                activities=["Fails"],
                estimated_time_hours=1.0
            )

        # Invalid study hours (negative)
        with self.assertRaises(ValidationError):
            RevisionDay(
                day=1,
                topics=["Invalid"],
                activities=["Fails"],
                estimated_time_hours=-1.0
            )

if __name__ == '__main__':
    unittest.main()
