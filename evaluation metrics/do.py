import requests

# 1️⃣ Read exam document from file
with open("question_paper.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2️⃣ Payload
payload = {
    "exam_id": "CS_ADV_2025",
    "raw_text": raw_text
}

# 3️⃣ Send request
response = requests.post(
    "http://localhost:8000/questions/bulk",
    json=payload
)

# 4️⃣ Output
print("Status:", response.status_code)

content_type = response.headers.get("content-type", "")
print("Content-Type:", content_type)

if content_type.startswith("application/json"):
    print(response.json())
else:
    print("Non-JSON response from server:")
    print(response.text)
