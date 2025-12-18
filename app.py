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
async def split_questions(raw_text: str):
    prompt = f"""
You are given an exam answer-key or question paper text.

Extract ALL questions.

Return STRICT JSON array:
[
  {{
    "question_number": 1,
    "question_text": "...",
    "max_marks": 10
  }}
]

Only valid JSON. No explanation.

TEXT:
{raw_text}
"""

    response = await asyncio.to_thread(
        ollama.generate,
        model="smollm2:latest",
        prompt=prompt.strip()
    )

    # --- SAFELY EXTRACT JSON ARRAY ---
    import re
    match = re.search(r"\[.*\]", response["response"], re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return JSON. Raw output:\n{response['response']}")
    
    questions_json = match.group(0)
    questions = json.loads(questions_json)
    return questions

# ----------------- LLM: CLASSIFY QUESTION -----------------
async def classify_question(question_text: str) -> str:
    prompt = f"""
Classify the exam question as exactly one word:
DESCRIPTIVE or TECHNICAL.

Question:
{question_text}
"""

    response = await asyncio.to_thread(
        ollama.generate,
        model="smollm2:latest",
        prompt=prompt.strip()
    )

    output = response["response"].upper()
    return "TECHNICAL" if "TECHNICAL" in output else "DESCRIPTIVE"

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
