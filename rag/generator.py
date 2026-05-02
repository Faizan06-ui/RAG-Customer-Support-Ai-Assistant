import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        # Load API key from environment
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not found in environment variables!")

        # Set OpenRouter endpoint and headers
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }

        # Set the model
        self.model = "mistralai/ministral-14b-2512"

    def generate_response(self, query, top_matches, top_k=3, history=[]):
        top_matches = top_matches[:top_k]
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in top_matches])

        prompt = f"""You are a helpful customer support assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

        messages = [
            {"role": "system", "content": (
                "You are a professional AI-powered customer support assistant. "
                "Use the retrieved context below to answer the user's question accurately and helpfully. "
                "Important rules:\n"
                "- Never address the user by any name from the retrieved context. Always say 'you' not a specific name.\n"
                "- Never mention that you are using retrieved examples or past cases.\n"
                "- Keep responses focused and professional.\n"
                "- If steps are needed, use numbered lists.\n"
                "- Be empathetic but concise."
            )},
            *history,
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as http_err:
            return f"❌ OpenRouter API error: {http_err.response.status_code}\n{http_err.response.text}"
        except Exception as e:
            return f"⚠️ Unexpected error: {e}"
