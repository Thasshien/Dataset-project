# app.py
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient
import ollama
import json

# ----------------- APP -----------------
app = FastAPI()

# ----------------- MONGODB -----------------
client = MongoClient("mongodb://localhost:27017")
db = client["tries_db"]
questions_collection = db["questions"]

# ----------------- REQUEST MODELS -----------------
class BulkQuestionRequest(BaseModel):
    exam_id: str
    raw_text: str

# ----------------- LLM: SPLIT QUESTIONS -----------------
# ----------------- LLM: SPLIT QUESTIONS -----------------
# ----------------- LLM: SPLIT QUESTIONS -----------------
async def split_questions(raw_text: str):
    prompt = f"""
You are given an exam answer-key or question paper text.

Extract ALL questions.

Return STRICT JSON array of objects:
[
  {{
    "question_number": 1,
    "question_text": "...",
    "max_marks": 10
  }}
]

- Each question must be a JSON object.
- Do NOT include any explanation or text outside the JSON array.
- Preserve all newlines inside question_text.
- Ensure JSON is valid.

TEXT:
{raw_text}
"""

    response = await asyncio.to_thread(
        ollama.generate,
        model="smollm2:latest",
        prompt=prompt.strip()
    )

    raw_output = response["response"]

    # --- SAFELY EXTRACT JSON ARRAY ---
    import re
    try:
        match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw_output}")
        questions_json = match.group(0)
        questions = json.loads(questions_json)
    except json.JSONDecodeError:
        # fallback: try splitting manually by QUESTION_x_START markers
        print("Warning: JSON parse failed, falling back to manual split...")
        splits = re.split(r"\[QUESTION_\d+_START\]", raw_text)
        questions = []
        for i, s in enumerate(splits[1:], start=1):
            # extract marks if available
            marks_match = re.search(r"\[(\d+)\s*marks\]", s)
            max_marks = int(marks_match.group(1)) if marks_match else 0
            questions.append({
                "question_number": i,
                "question_text": s.strip(),
                "max_marks": max_marks
            })

    return questions

# ----------------- BULK INGEST -----------------
@app.post("/questions/bulk")
async def ingest_questions(payload: BulkQuestionRequest):

    # 1️⃣ Split questions
    questions = await split_questions(payload.raw_text)

    inserted_ids = []

    for q in questions:
        q_type = await classify_question(q["question_text"])

        doc = {
            "exam_id": payload.exam_id,
            "question_number": q["question_number"],
            "question_text": q["question_text"],
            "question_type": q_type,
            "max_marks": q["max_marks"],
            "created_at": datetime.utcnow()
        }

        # 2️⃣ Attach schema by type
        if q_type == "DESCRIPTIVE":
            doc["rubric"] = {
                "traits": [
                    {"name": "Concept Coverage", "weight": 0.4},
                    {"name": "Examples / Application", "weight": 0.3},
                    {"name": "Organization & Clarity", "weight": 0.3}
                ]
            }

        else:  # TECHNICAL
            doc.update({
                "model_answer": "",
                "keywords": [],
                "solution_chunks": [],
                "numeric_rules": {
                    "tolerance": 0.05,
                    "expected_numbers": []
                }
            })

        result = questions_collection.insert_one(doc)
        inserted_ids.append(str(result.inserted_id))

    return {
        "message": "Questions extracted & stored",
        "total_questions": len(inserted_ids),
        "question_ids": inserted_ids
    }
