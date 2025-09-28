# test_groq_call.py
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
print("Using base:", base)
print("GROQ key present:", bool(key))

client = OpenAI(api_key=key, base_url=base)

# pick a model you saw in list_groq_models.py (e.g. "llama-3.3-70b-versatile")
model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":"Say hello in one sentence."}],
        max_tokens=30
    )
    print("Reply:", resp.choices[0].message.content)
except Exception as e:
    print("Model test failed:", type(e), e)
