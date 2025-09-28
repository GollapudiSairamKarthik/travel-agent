from dotenv import load_dotenv
import os
from openai import OpenAI

# Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
model = os.getenv("LLM_MODEL")

print("Using base:", api_base)
print("Using model:", model)

client = OpenAI(api_key=api_key, base_url=api_base)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello Groq! Introduce yourself in one line."}],
)

print("✅ Groq reply:", resp.choices[0].message.content)
