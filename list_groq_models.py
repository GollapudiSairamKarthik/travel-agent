# list_groq_models.py
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")

print("Using base:", api_base)
client = OpenAI(api_key=api_key, base_url=api_base)

# List available models for your account/endpoint
try:
    models = client.models.list()
    print("Available models (first 40):")
    for m in models.data[:40]:
        print("-", m.id)
except Exception as e:
    print("Failed to list models:", repr(e))
