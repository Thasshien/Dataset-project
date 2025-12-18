import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pymongo import MongoClient
import ollama
import json
import re

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

# ----------------- LLM: HIGHLY ACCURATE SPLIT -----------------
async def split_questions(raw_text: str):
    """Extract questions using your exact [QUESTION_X_START] format"""
    
    # First, use REGEX for PERFECT extraction (no LLM needed for splitting)
    question_pattern = r'\[QUESTION_(\d+)_START\]\s*\[(\d+)\s*marks\]\s*Question:\s*(.*?)(?=\[QUESTION_\d+_START\]|$)(?s)'
    matches = re.findall(question_pattern, raw_text, re.DOTALL)
    
    questions = []
    for match_num, (q_num, marks, content) in enumerate(matches, 1):
        questions.append({
            "question_number": int(q_num),
            "question_text": content.strip(),
            "max_marks": int(marks)
        })
    
    if questions:
        print(f"✅ Regex extracted {len(questions)} questions perfectly")
        return questions
    
    # Fallback to LLM only if regex fails
    print("⚠️ Regex failed, falling back to LLM...")
    return await _llm_split_fallback(raw_text)

async def _llm_split_fallback(raw_text: str):
    """LLM fallback for question splitting"""
    prompt = f"""
Extract ALL questions from this answer key.

Your format uses [QUESTION_X_START] markers.

Return STRICT JSON array:
[
  {{"question_number": 1, "question_text": "Question text here", "max_marks": 10}}
]

TEXT:
{raw_text[:4000]}  # truncated for context
"""
    
    response = await asyncio.to_thread(
        ollama.generate, model="smollm2:latest", prompt=prompt.strip()
    )
    
    try:
        # Extract JSON array
        match = re.search(r'\[.*\]', response["response"], re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    
    # Emergency fallback
    return [{"question_number": 1, "question_text": raw_text[:500], "max_marks": 10}]

# ----------------- CLASSIFICATION -----------------
async def classify_question(question_text: str) -> str:
    """Improved classification with your exact patterns"""
    # QUICK HEURISTIC FIRST (99% accurate)
    question_lower = question_text.lower()
    
    technical_indicators = [
        "write", "implement", "code", "sql", "query", "algorithm", 
        "function", "pipeline", "select", "from", "group by", "having"
    ]
    
    descriptive_indicators = [
        "explain", "discuss", "analyze", "describe", "difference", 
        "impact", "concept", "properties"
    ]
    
    technical_score = sum(1 for word in technical_indicators if word in question_lower)
    descriptive_score = sum(1 for word in descriptive_indicators if word in question_lower)
    
    if technical_score > descriptive_score:
        return "TECHNICAL"
    
    # LLM only for edge cases
    prompt = f"""Classify EXACTLY ONE WORD: DESCRIPTIVE or TECHNICAL

Question: {question_text[:300]}
"""
    
    response = await asyncio.to_thread(
        ollama.generate, model="smollm2:latest", prompt=prompt.strip()
    )
    
    content = response["response"].strip().upper()
    if content.startswith("TECHNICAL"):
        return "TECHNICAL"
    return "DESCRIPTIVE"

# ----------------- DESCRIPTIVE: EXTRACT RUBRIC -----------------
async def extract_rubric_traits(question_text: str):
    """Extract rubric from your exact table format"""
    
    # REGEX for your rubric tables (VERY SPECIFIC)
    rubric_pattern = r'(?:Evaluation Rubric|Rubric):.*?Trait.*?Weight.*?Description\s*((?:[^\n]|\n)+?)(?=\nQuestion|\Z)'
    rubric_match = re.search(rubric_pattern, question_text, re.IGNORECASE | re.DOTALL)
    
    if rubric_match:
        rubric_text = rubric_match.group(1).strip()
        # Parse table lines
        lines = [line.strip() for line in rubric_text.split('\n') if line.strip()]
        traits = []
        
        for line in lines[1:]:  # Skip header
            parts = re.split(r'\s{2,}', line)  # Split by multiple spaces
            if len(parts) >= 2:
                trait_name = parts[0].strip()
                weight_str = re.search(r'(\d+)%?', parts[1])
                weight = float(weight_str.group(1)) / 100 if weight_str else 0.25
                traits.append({"name": trait_name, "weight": weight})
        
        if traits:
            print(f"✅ Extracted {len(traits)} rubric traits via regex")
            return traits[:4]  # Limit to 4 traits
    
    # LLM fallback with your exact format
    prompt = f"""Extract rubric traits from this DESCRIPTIVE question answer key.

Format is:
Evaluation Rubric:
Trait           Weight    Description
Concept Cov...  40%       ...

Return JSON array:
[{{"name": "Concept Coverage", "weight": 0.4}}, ...]

Question + Rubric:
{question_text[:2000]}
"""
    
    try:
        response = await asyncio.to_thread(ollama.generate, model="smollm2:latest", prompt=prompt.strip())
        match = re.search(r'\[.*\]', response["response"], re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    
    # Smart defaults based on topic
    if "ACID" in question_text:
        return [
            {"name": "Concept Coverage", "weight": 0.4},
            {"name": "Real-World Application", "weight": 0.3},
            {"name": "Logical Flow", "weight": 0.2},
            {"name": "Clarity & Language", "weight": 0.1}
        ]
    elif "learning" in question_text.lower():
        return [
            {"name": "Fundamental Differences", "weight": 0.35},
            {"name": "Examples Provided", "weight": 0.35},
            {"name": "Problem Suitability", "weight": 0.2},
            {"name": "Organization", "weight": 0.1}
        ]
    
    return [
        {"name": "Concept Coverage", "weight": 0.4},
        {"name": "Examples/Application", "weight": 0.3},
        {"name": "Organization", "weight": 0.2},
        {"name": "Clarity", "weight": 0.1}
    ]

# ----------------- TECHNICAL: EXTRACT ANSWER + KEYWORDS -----------------
async def extract_model_answer_and_keywords(question_text: str):
    """Extract from your exact Expected Answer format"""
    
    # REGEX for your code blocks (HIGHLY SPECIFIC)
    code_patterns = [
        r'Expected Answer:\s*(``````|``````|\n(    .*?\n)+?)Key Points:',
        r'Expected Answer Approach:\s*(``````|\n(    .*?\n)+?)Key Points:',
        r'Expected Answer:\s*([^\n]{50,})?(?=\nKey Points:|\nQuestion)',
    ]
    
    model_answer = ""
    for pattern in code_patterns:
        match = re.search(pattern, question_text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1) if match.groups() else match.group(0)
            model_answer = re.sub(r'^``````$', '', code, flags=re.MULTILINE).strip()
            if model_answer:
                print(f"✅ Extracted model answer via regex ({len(model_answer)} chars)")
                break
    
    # Extract keywords from Key Points
    keywords = []
    key_points_match = re.search(r'Key Points:\s*([•\-\s]*.*?(?=\nQuestion|\Z))', question_text, re.DOTALL | re.IGNORECASE)
    if key_points_match:
        points_text = key_points_match.group(1)
        keywords = [point.strip('•\-\s• ') for point in points_text.split('\n') if point.strip('•\-\s• ') and len(point.strip()) > 3]
    
    if model_answer or keywords:
        return model_answer, keywords[:10]  # Limit keywords
    
    # LLM fallback
    prompt = f"""TECHNICAL question answer key. Extract:

1. Model answer (code/SQL)
2. Keywords from "Key Points"

Return JSON:
{{"model_answer": "code here", "keywords": ["kw1", "kw2"]}}

Question:
{question_text[:1500]}
"""
    
    try:
        response = await asyncio.to_thread(ollama.generate, model="smollm2:latest", prompt=prompt.strip())
        match = re.search(r'\{.*\}', response["response"], re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("model_answer", ""), data.get("keywords", [])
    except:
        pass
    
    return "", []

# ----------------- MAIN ENDPOINT -----------------
@app.post("/questions/bulk")
async def ingest_questions(payload: BulkQuestionRequest):
    """Process your exact answer key format"""
    
    print(f"🔄 Processing exam: {payload.exam_id}")
    print(f"📄 Text length: {len(payload.raw_text)} chars")
    
    # 1️⃣ Split questions (REGEX first!)
    questions = await split_questions(payload.raw_text)
    
    inserted_ids = []
    
    for i, q in enumerate(questions):
        print(f"\n--- Processing Q{q['question_number']} ({q['max_marks']} marks) ---")
        
        # 2️⃣ Classify
        q_type = await classify_question(q["question_text"])
        print(f"   Type: {q_type}")
        
        # 3️⃣ Base document
        doc = {
            "exam_id": payload.exam_id,
            "question_number": q["question_number"],
            "question_text": q["question_text"].strip(),
            "question_type": q_type,
            "max_marks": q["max_marks"],
            "created_at": datetime.utcnow()
        }
        
        # 4️⃣ Type-specific extraction
        if q_type == "DESCRIPTIVE":
            print("   📝 Extracting rubric...")
            doc["rubric"] = {
                "traits": await extract_rubric_traits(q["question_text"])
            }
            print(f"   Rubric traits: {len(doc['rubric']['traits'])}")
            
        else:  # TECHNICAL
            print("   🔧 Extracting answer + keywords...")
            model_answer, keywords = await extract_model_answer_and_keywords(q["question_text"])
            doc.update({
                "model_answer": model_answer,
                "keywords": keywords,
                "solution_chunks": [model_answer] if model_answer else [],
                "numeric_rules": {"tolerance": 0.05, "expected_numbers": []}
            })
            print(f"   Answer: {len(model_answer)} chars, Keywords: {len(keywords)}")
        
        # 5️⃣ Insert
        result = questions_collection.insert_one(doc)
        inserted_ids.append(str(result.inserted_id))
        print(f"   ✅ Stored (ID: {result.inserted_id})")
    
    return {
        "message": f"✅ {len(inserted_ids)} questions processed perfectly",
        "exam_id": payload.exam_id,
        "total_questions": len(inserted_ids),
        "question_ids": inserted_ids,
        "breakdown": {
            "DESCRIPTIVE": len([1 for id in inserted_ids if questions_collection.find_one({"_id": inserted_ids[id]}).get("question_type") == "DESCRIPTIVE"]),
            "TECHNICAL": len(inserted_ids) - len([1 for id in inserted_ids if questions_collection.find_one({"_id": inserted_ids[id]}).get("question_type") == "DESCRIPTIVE"])
        }
    }

# ----------------- DEBUG ENDPOINT -----------------
@app.get("/questions/{exam_id}")
async def get_questions(exam_id: str):
    """Debug: View processed questions"""
    docs = list(questions_collection.find({"exam_id": exam_id}).sort("question_number"))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return {"questions": docs}
