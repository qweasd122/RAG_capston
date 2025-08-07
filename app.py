from flask import Flask, request, Response, after_this_request
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
CORS(app, resources={r"/*": {"origins": "https://test.choeaecafe.com/qa_page.php"}})

# 모델 준비
model = SentenceTransformer("all-MiniLM-L6-v2")

# 텍스트 추출
def load_text_chunks():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item["tweet_text"] for item in data["documents"] if "tweet_text" in item]
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
        response = Response('', status=204)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response
    
    @after_this_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    messages = request.json.get("messages", [])

    user_query = request.json.get("query", "")
    if not user_query:
        return {"error": "질문이 비어 있어요…"}, 400

    query_emb = model.encode([user_query])
    top_k = 5
    _, indices = index.search(query_emb, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]


    context = "\n\n".join(retrieved_chunks)
    prompt = prompt = f"""
당신은 아이돌 팬들을 위해 생일 카페 정보를 친절하고 정성스럽게 알려주는 AI입니다.
아래는 문서에서 추출한 생일 카페 관련 정보들이에요:

{context}

사용자의 질문은 다음과 같아요:
"{user_query}"

위 정보를 참고하여, 사용자에게 도움이 되는 생일 카페 정보나 추천을 따뜻한 말투로 안내해주세요.
가능하면 구체적인 날짜, 장소, 이벤트 특징도 알려주세요.
문서에 없는 내용은 지어내지 않고, 솔직하게 없다고 말하세요.
문서 내용 중 웹 링크 형식의 내용은 사용자가 직접 요구할 때만 말하세요.
답변:
"""

    def stream_response():
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="deepseek-chat",
            # messages=[{"role": "system", "content": "아이돌 생일카페 정보를 모아 친절하게 답변해드립니다."},
            #          {"role": "user", "content": prompt}],
            messages=messages,
            stream=True
        )
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content


    # response = Response(stream_response(), content_type='text/plain')
    # response.headers["Access-Control-Allow-Origin"] = "*"
    # response.headers["Access-Control-Allow-Headers"] = "*"
    # response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    # return response
    return Response(stream_response(), content_type='text/plain')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
