from flask import Flask, request, Response
from flask_cors import CORS
import json
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import os

# 준비된 JSON 문서 경로
JSON_PATH = "events.json"

# 서버 준비
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 모델 준비
model = SentenceTransformer("all-MiniLM-L6-v2")

# 텍스트 추출
def load_text_chunks():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["content"] for item in data["documents"]]
    text = " ".join(texts)
    words = text.split()
    chunks = [' '.join(words[i:i+300]) for i in range(0, len(words), 300)]
    return chunks

chunks = load_text_chunks()
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

# LLM 클라이언트
client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.deepseek.com"
)

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    if request.method == "OPTIONS":
        return '', 204
    
    user_query = request.json.get("query", "")
    if not user_query:
        return {"error": "질문이 비어 있어요…"}, 400

    query_emb = model.encode([user_query])
    top_k = 5
    _, indices = index.search(query_emb, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]


    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
다음은 문서에서 추출된 내용이에요:

{context}

질문: {user_query}
답변:"""

    def stream_response():
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "아이돌 생일카페 정보를 모아 친절하게 답변해드립니다."},
                     {"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content

    return Response(stream_response(), content_type='text/plain')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
