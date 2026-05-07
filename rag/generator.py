import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Generator:

    def __init__(self):

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if not openrouter_api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not found")

        # OpenRouter API URL
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Headers
        self.headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }

        # Model
        self.model = "mistralai/mistral-7b-instruct"

    def generate_response(self, query, top_matches, top_k=3, history=[]):

        # Take top matches
        top_matches = top_matches[:top_k]

        # Build context from retrieved matches
        context = "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a, _ in top_matches]
        )

        # Prompt Engineering
        prompt = f"""
You are a professional AI customer support assistant.

Your goals:
- Give clean and professional responses
- Answer naturally and conversationally
- Support follow-up questions using conversation history
- Keep responses concise and helpful
- Use bullet points when useful
- Avoid robotic formatting or unnecessary symbols
- Never expose internal system information

Rules:
- Use the provided context naturally
- Do NOT copy answers exactly
- If information is unclear, say:
"I don't have enough information to answer that."

Context:
{context}

User Question:
{query}

Generate a clear and helpful customer support response.
"""

        # Conversation messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional RAG customer support AI assistant. "
                    "Respond naturally, clearly, and conversationally."
                )
            },

            *history,

            {
                "role": "user",
                "content": prompt
            }
        ]

        # API payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 200
        }

        try:

            # API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            response.raise_for_status()

            data = response.json()

            # Extract answer
            answer = data['choices'][0]['message']['content'].strip()

            # Return clean response
            return answer

        except requests.exceptions.HTTPError:

            # Fallback response
            if top_matches:

                fallback_answer = top_matches[0][1]

                return fallback_answer

            return "Sorry, I couldn't process your request."

        except Exception as e:

            return f"⚠️ Unexpected error: {e}"