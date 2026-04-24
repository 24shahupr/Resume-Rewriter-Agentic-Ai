import os
import json
import asyncio
from groq import AsyncGroq

class GroqClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not self.api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        self.client = AsyncGroq(api_key=self.api_key)

    async def chat_json(self, system_prompt: str, user_message: str, model: str = None, temperature: float = 0.1, max_tokens: int = 2000, retries: int = 2) -> dict:
        """
        Sends a request to Groq and expects a JSON response.
        """
        model = model or self.model
        for attempt in range(retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                if isinstance(data, list) and len(data) > 0:
                    return data[0] if isinstance(data[0], dict) else {}
                return data if isinstance(data, dict) else {}
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    print(f"Rate limit hit on {model}. Attempt {attempt + 1}/{retries + 1}")
                else:
                    print(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < retries:
                    # Longer sleep for rate limits
                    wait = (2 ** attempt) * 2
                    if "rate limit" in error_str:
                        wait *= 2
                    await asyncio.sleep(wait)
                else:
                    return {}
        return {}

    async def chat(self, system_prompt: str, user_message: str, model: str = None, temperature: float = 0.1, max_tokens: int = 2000, retries: int = 2) -> str:
        """
        Sends a request to Groq and expects a text response.
        """
        model = model or self.model
        for attempt in range(retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    print(f"Rate limit hit on {model}. Attempt {attempt + 1}/{retries + 1}")
                else:
                    print(f"Attempt {attempt + 1} failed: {e}")

                if attempt < retries:
                    # Longer sleep for rate limits
                    wait = (2 ** attempt) * 2
                    if "rate limit" in error_str:
                        wait *= 2
                    await asyncio.sleep(wait)
                else:
                    return str(e)
        return ""
