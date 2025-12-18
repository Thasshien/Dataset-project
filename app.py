# app.py
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pymongo import MongoClient
import ollama

# ----------------- APP -----------------
app = FastAPI()

# ----------------- MONGODB -----------------
client = MongoClient("mongodb://localhost:27017")
db = client["tries_db"]
questions_collection = db["questions"]

# ----------------- REQUEST MODEL -----------------
class QuestionRequest(BaseModel):
    exam_id: str
    question_number: int
    question_text: str
    max_marks: int
    question_type: Optional[str] = None  # classified if missing

# ----------------- QUESTION CLASSIFIER -----------------
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

    content = response["response"].strip().upper()

    if "TECHNICAL" in content:
        return "TECHNICAL"
    return "DESCRIPTIVE"

# ----------------- CREATE QUESTION -----------------
@app.post("/questions")
async def create_question(payload: QuestionRequest):
    # 1️⃣ Classify
    q_type = payload.question_type
    if not q_type:
        q_type = await classify_question(payload.question_text)

    # 2️⃣ Base document
    question_doc = {
        "exam_id": payload.exam_id,
        "question_number": payload.question_number,
        "question_text": payload.question_text,
        "question_type": q_type,
        "max_marks": payload.max_marks,
        "created_at": datetime.utcnow()
    }

    # 3️⃣ Populate based on type
    if q_type == "DESCRIPTIVE":
        question_doc["rubric"] = {
            "traits": [
                { "name": "Concept Coverage", "weight": 0.4 },
                { "name": "Logical Flow", "weight": 0.3 },
                { "name": "Clarity", "weight": 0.3 }
            ]
        }

    elif q_type == "TECHNICAL":
        question_doc.update({
            "model_answer": "",
            "keywords": [],
            "solution_chunks": [],
            "numeric_rules": {
                "tolerance": 0.05,
                "expected_numbers": []
            }
        })

    # 4️⃣ Store in MongoDB
    result = questions_collection.insert_one(question_doc)

    # 5️⃣ Response
    return {
        "message": "Question stored successfully",
        "question_id": str(result.inserted_id),
        "question_type": q_type
    }
