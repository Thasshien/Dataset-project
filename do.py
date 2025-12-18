import requests

# 1. Read raw question from local text file
with open("question.txt", "r") as f:
    question_text = f.read().strip()

# 2. Prepare payload
payload = {
    "exam_id": "EXAM123",
    "question_number": 1,
    "question_text": question_text,
    "max_marks": 10
    # Note: question_type is NOT provided → Ollama will classify
}

# 3. POST to FastAPI
url = "http://localhost:8000/questions"
response = requests.post(url, json=payload)

# 4. Print response
print(response.json())
