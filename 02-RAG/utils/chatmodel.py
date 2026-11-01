"""Step 5 of RAG: send chat messages to a Gemini model and get the answer back."""
import os
from typing import Dict, List

from google import genai
from google.genai import errors, types

from .embedding import get_client


class ChatGemini:
    def __init__(self, model_name: str = None, client: genai.Client = None):
        # Flash-Lite is fast and has a roomier free-tier daily quota than Flash.
        # Free-tier limits are counted per model, so switching models also helps when one runs out.
        self.model_name = model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
        self.client = client or get_client()

    def run(self, messages: List[Dict[str, str]]) -> str:
        """messages: [{"role": "system" | "user" | "assistant", "content": "..."}]"""
        # Gemini takes system messages as a separate instruction, and calls the
        # assistant role "model".
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system or None),
            )
        except errors.ClientError as e:
            if e.code == 429:
                raise RuntimeError(
                    f"Free-tier quota for {self.model_name} is used up. Wait for the reset "
                    "(daily quotas reset at midnight Pacific time) or set GEMINI_CHAT_MODEL "
                    "to another model, e.g. gemini-3.1-flash-lite. "
                    "See your limits at https://aistudio.google.com/rate-limit"
                ) from e
            raise
        return response.text


if __name__ == "__main__":
    chat = ChatGemini()
    print(chat.run([
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "What is a vector database?"},
    ]))
