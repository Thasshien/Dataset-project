import re
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import ollama

# -------------------- APP --------------------
app = FastAPI()

# -------------------- DB --------------------
client = MongoClient("mongodb+srv://daktrboys05_db_user:gdgclubproject@to-do-list.qmqixqe.mongodb.net/")
db = client["tries_db"]
questions_collection = db["questions"]

# -------------------- REQUEST MODEL --------------------
class BulkQuestionRequest(BaseModel):
    exam_id: str
    raw_text: str

# -------------------- UTIL --------------------
def safe_json(text: str):
    """
    NEVER throws.
    Attempts to extract JSON, otherwise returns {}.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}

        raw = match.group(0)

        # First attempt
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Second attempt: normalize quotes
        cleaned = raw.replace("'", '"')
        cleaned = cleaned.replace("\n", " ")
        cleaned = cleaned.replace("\t", " ")

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Final fallback
        print("⚠️ LLM returned invalid JSON. Skipping enrichment.")
        return {}

    except Exception:
        return {}

# -------------------- SPLIT QUESTIONS --------------------
def split_questions(raw_text: str):
    """
    Guaranteed splitter for your QUESTION_X_START format.
    """
    blocks = re.split(r"\[QUESTION_(\d+)_START\]", raw_text)
    questions = []

    for i in range(1, len(blocks), 2):
        q_no = int(blocks[i])
        content = blocks[i + 1]

        marks = 0
        m = re.search(r"\[(\d+)\s*marks\]", content, re.I)
        if m:
            marks = int(m.group(1))

        q_match = re.search(r"Question:\s*(.*)", content, re.DOTALL)
        q_text = q_match.group(1).strip() if q_match else content.strip()

        questions.append({
            "question_number": q_no,
            "question_text": q_text,
            "max_marks": marks,
            "full_block": content.strip()
        })

    return questions

# -------------------- CLASSIFY --------------------
TECHNICAL_HINTS = [
    "write", "query", "sql", "algorithm",
    "pipeline", "implement", "code"
]

def classify_question(text: str):
    return "TECHNICAL" if any(k in text.lower() for k in TECHNICAL_HINTS) else "DESCRIPTIVE"

# -------------------- PROMPTS --------------------
DESCRIPTIVE_PROMPT = """
You are generating grading metadata for an exam.

TASK:
Create rubric-based grading configuration.

RULES:
- Output STRICT JSON only
- Do NOT explain anything
- Do NOT omit fields

FORMAT:
{
  "descriptive_config": {
    "rubric": [
      {
        "trait": "Concept Coverage",
        "weight": 0.4,
        "description": "Coverage of all core concepts"
      }
    ],
    "scoring_method": "rubric_based",
    "feedback_style": "conceptual"
  }
}

QUESTION:
<<<QUESTION_TEXT>>>
"""

TECHNICAL_PROMPT = """
You are generating grading metadata for an exam.

TASK:
Extract grading components from a technical question.

RULES:
- Output STRICT JSON only
- model_answer must be complete
- keywords >= 5

FORMAT:
{
  "technical_config": {
    "model_answer": "Expected solution / code / query",
    "solution_chunks": [
      "key step 1",
      "key step 2"
    ],
    "keywords": [
      "KEYWORD1",
      "KEYWORD2"
    ],
    "numeric_rules": {
      "tolerance": 0.05,
      "expected_numbers": []
    },
    "scoring_method": "rag_keyword_numeric"
  }
}

QUESTION + EXPECTED ANSWER:
<<<QUESTION_BLOCK>>>
"""

# -------------------- API --------------------
@app.post("/questions/bulk")
def ingest_questions(payload: BulkQuestionRequest):

    questions = split_questions(payload.raw_text)

    print(f"\n✅ Detected {len(questions)} questions")

    inserted_ids = []

    for q in questions:
        q_type = classify_question(q["question_text"])

        base_doc = {
            "exam_id": payload.exam_id,
            "question_number": q["question_number"],
            "question_text": q["question_text"],
            "question_type": q_type,
            "max_marks": q["max_marks"],
            "created_at": datetime.utcnow()
        }

        print(f"\n--- Q{q['question_number']} ({q_type}) ---")

        if q_type == "DESCRIPTIVE":
            prompt = DESCRIPTIVE_PROMPT.replace(
                "<<<QUESTION_TEXT>>>", q["question_text"]
            )
            resp = ollama.generate(model="llama3:latest", prompt=prompt)
            base_doc.update(safe_json(resp["response"]))

        else:
            prompt = TECHNICAL_PROMPT.replace(
                "<<<QUESTION_BLOCK>>>", q["full_block"]
            )
            resp = ollama.generate(model="llama3:latest", prompt=prompt)
            base_doc.update(safe_json(resp["response"]))

        result = questions_collection.insert_one(base_doc)
        inserted_ids.append(str(result.inserted_id))

    return {
        "message": "Questions ingested successfully",
        "total_questions": len(inserted_ids),
        "question_ids": inserted_ids
    }
