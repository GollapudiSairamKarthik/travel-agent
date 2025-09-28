# print_env.py — run this locally, do NOT paste key here
import os
from dotenv import load_dotenv

load_dotenv()

g = os.getenv("GROQ_API_KEY")
o = os.getenv("OPENAI_API_KEY")
b = os.getenv("OPENAI_API_BASE")

print("GROQ_API_KEY present:", bool(g))
print("GROQ_API_KEY length:", len(g) if g else None)
print("OPENAI_API_KEY present:", bool(o))
print("OPENAI_API_BASE:", b)
# optional debug (shows quotes so you can spot stray spaces/newlines)
print("GROQ_API_KEY repr (for debug):", repr(g) if g else None)
