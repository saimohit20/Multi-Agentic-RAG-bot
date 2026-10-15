"""LLM prompt templates for question generation."""

QA_GENERATION_PROMPT = """You are an expert question generator for RAG system evaluation datasets.

Given the following training material content, generate exactly 5 questions.

CONTENT:
{content}

CRITICAL RULES:
1. Write questions as a learner or employee would type into a search or chat system.
2. NEVER reference the document, slide, video, presentation, or training material in the question.
3. NEVER use phrases like "according to the slide", "in the video", "mentioned in the training", "in the presentation".
4. Questions must test knowledge of the topic, not knowledge of the document.
5. Each question must be self-contained and understandable without seeing the source material.
6. Ground truth answers should be clear and concise (1-3 sentences), based on the content.
7. Extract 2-3 relevant domain keywords per question.
8. Assign one answer_type per question from the allowed values.
9. Assign a short subtopic label (the specific concept being tested).

BAD example (do NOT do this):
  "What does the slide about Industry 4.0 define?"
  "What does the 'Visions for the Future' slide highlight?"

GOOD example (do this):
  "What is Industry 4.0 and how does it relate to digital transformation?"
  "What are the main fears workers have about Industry 4.0?"

QUESTION TYPES (use exactly one per question, one of each across the 5):
1. factual    - Direct recall of a fact, definition, or specific data point
2. conceptual - Understanding of a concept, relationship, or principle
3. procedural - How something works or the steps in a process
4. application - How a concept is applied in a real-world scenario
5. summary    - Synthesis of multiple related ideas into one answer

RESPONSE FORMAT:
Return a JSON array with exactly 5 objects. No other text.
{{
    "question": "question text (no document references)",
    "ground_truth": "correct answer in 1-3 sentences",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "answer_type": "factual|conceptual|procedural|application|summary",
    "subtopic": "short concept label"
}}
"""
