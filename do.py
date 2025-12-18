# do.py
import requests

# 1️⃣ Read raw exam text
with open("question_paper.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2️⃣ Payload
payload = {
    "exam_id": "CS_ADV_2025",
    "raw_text": raw_text
}

# 3️⃣ Send to API
url = "http://localhost:8000/questions/bulk"
response = requests.post(url, json=payload)

# 4️⃣ Output
print(response.status_code)
print(response.json())
