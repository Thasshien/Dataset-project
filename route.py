# app.py
import os
import asyncio
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import motor.motor_asyncio
import ollama  # pip install ollama


app = FastAPI()

# ---------- Mongo setup ----------
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client["exam_grader"]
questions_collection = db["questions"]


# ---------- Pydantic models ----------

class RubricTrait(BaseModel):
    name: str
    weight: float


class NumericRules(BaseModel):
    tolerance: float = 0.05
    expected_numbers: List[float] = Field(default_factory=list)


class QuestionCreate(BaseModel):
    exam_id: str
    question_number: int
    question_text: str
    max_marks: int

    question_type: Optional[Literal["DESCRIPTIVE", "TECHNICAL"]] = None
    rubric: Optional[Dict[str, List[RubricTrait]]] = None
    model_answer: Optional[str] = None
    keywords: Optional[List[str]] = None
    solution_chunks: Optional[List[str]] = None
    numeric_rules: Optional[NumericRules] = None


class QuestionResponse(QuestionCreate):
    id: str


# ---------- Helper: classify with Ollama ----------

async def classify_question_type(
    question_text: str,
    model_answer: Optional[str]
) -> Literal["DESCRIPTIVE", "TECHNICAL"]:

    prompt = f"""
You are an exam-question classifier.

Classify the question as exactly one word:
DESCRIPTIVE or TECHNICAL.

Question:
{question_text}

Solution:
{model_answer or "N/A"}
"""

    # Run blocking Ollama call in a thread
    response = await asyncio.to_thread(
        ollama.generate,
        model="llama3.1",
        prompt=prompt.strip()
    )

    content = response["response"].strip().upper()

    if content.startswith("TECHNICAL"):
        return "TECHNICAL"
    if content.startswith("DESCRIPTIVE"):
        return "DESCRIPTIVE"

    raise ValueError(f"Invalid classifier output: {content}")


# ---------- POST endpoint ----------

@app.post("/questions", response_model=QuestionResponse)
async def create_question(payload: QuestionCreate):

    # 1. Decide question type
    if payload.question_type:
        q_type = payload.question_type
    else:
        q_type = await classify_question_type(
            payload.question_text,
            payload.model_answer
        )

    # 2. Base document
    doc: Dict[str, Any] = {
        "exam_id": payload.exam_id,
        "question_number": payload.question_number,
        "question_text": payload.question_text,
        "question_type": q_type,
        "max_marks": payload.max_marks,
        "model_answer": payload.model_answer,
        "keywords": payload.keywords,
        "solution_chunks": payload.solution_chunks or [],
        "numeric_rules": (
            payload.numeric_rules.dict()
            if payload.numeric_rules
            else {"tolerance": 0.05, "expected_numbers": []}
        ),
        "created_by": "teacher_id",  # replace via auth later
        "created_at": datetime.utcnow(),
    }

    # 3. Rubric handling
    if q_type == "DESCRIPTIVE":
        doc["rubric"] = payload.rubric or {
            "traits": [
                {"name": "Concept Coverage", "weight": 0.4},
                {"name": "Logical Flow", "weight": 0.3},
                {"name": "Clarity", "weight": 0.3},
            ]
        }
    else:
        doc["rubric"] = None

    # 4. Insert into Mongo
    result = await questions_collection.insert_one(doc)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to insert question")

    # 5. Response
    return {
        **payload.dict(),
        "id": str(result.inserted_id),
        "question_type": q_type,
    }
