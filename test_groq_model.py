# test_groq_model.py
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # change in .env or set env var before running

print("Testing model:", model)
client = OpenAI(api_key=api_key, base_url=api_base)

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":"Say hello in one short line and state the model name."}],
    )
    print("Reply:", resp.choices[0].message.content)
except Exception as e:
    print("Model test failed:", repr(e))
