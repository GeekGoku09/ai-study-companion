import unittest
from pydantic import ValidationError
from src.schemas import DocumentChunk, MCQ, MCQList, TopicSummary, RevisionDay, RevisionPlan

class TestSchemas(unittest.TestCase):
    def test_document_chunk_validation(self):
        with self.assertRaises(ValidationError):
            DocumentChunk(content="   ", metadata={})
        chunk = DocumentChunk(content="Valid content", metadata={"page": 1})
        self.assertEqual(chunk.content, "Valid content")

    def test_mcq_validation(self):
        # Valid MCQ with exact answer match
        mcq = MCQ(
            question="What is entropy?",
            options=["Order", "Measure of disorder", "Heat only", "Pressure"],
            correct_answer="Measure of disorder",
            explanation="Entropy represents disorder."
        )
        self.assertEqual(mcq.correct_answer, "Measure of disorder")

        # Letter normalization ('B')
        mcq_letter = MCQ(
            question="What is entropy?",
            options=["Order", "Measure of disorder", "Heat only", "Pressure"],
            correct_answer="B",
            explanation="Entropy represents disorder."
        )
        self.assertEqual(mcq_letter.correct_answer, "Measure of disorder")

        # Invalid: correct answer not in options
        with self.assertRaises(ValidationError):
            MCQ(
                question="What is entropy?",
                options=["Order", "Measure of disorder", "Heat only", "Pressure"],
                correct_answer="Volume",
                explanation="None"
            )

        # Invalid: duplicate options
        with self.assertRaises(ValidationError):
            MCQ(
                question="What is entropy?",
                options=["Same", "Same", "Three", "Four"],
                correct_answer="Same",
                explanation="None"
            )

    def test_topic_summary_validation(self):
        with self.assertRaises(ValidationError):
            TopicSummary(topic="", summary="Summary")
        with self.assertRaises(ValidationError):
            TopicSummary(topic="Topic", summary="  ")

    def test_revision_day_validation(self):
        with self.assertRaises(ValidationError):
            RevisionDay(day=0, topics=["T1"], activities=["A1"], estimated_time_hours=1.0)
        with self.assertRaises(ValidationError):
            RevisionDay(day=1, topics=["T1"], activities=["A1"], estimated_time_hours=0.0)

if __name__ == "__main__":
    unittest.main()
