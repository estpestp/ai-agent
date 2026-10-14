import os
import requests

API_URL = os.getenv("AI_AGENT_URL", "https://ai-agent-api-zkjs.onrender.com")
API_TOKEN = os.getenv("API_TOKEN", "")

def ask_ai(message: str, page: dict | None = None) -> dict:
    if not API_TOKEN:
        raise RuntimeError("환경변수 API_TOKEN을 설정해 줘.")
    response = requests.post(
        f"{API_URL.rstrip('/')}/api/ask",
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        json={"message": message, "page": page or {}},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    print("AI Agent 서버 연결 테스트")
    question = input("질문 입력: ").strip()
    try:
        result = ask_ai(question, {"title": "테스트 페이지", "url": "", "summary": ""})
        print("\n서버 응답:")
        print(result.get("answer") or result)
    except requests.HTTPError as exc:
        print("서버 오류:", exc.response.status_code, exc.response.text)
    except requests.RequestException as exc:
        print("연결 오류:", exc)
    except RuntimeError as exc:
        print(exc)
