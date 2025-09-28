# llm_test.py
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY present:", bool(api_key))

# Create a client with your API key
client = OpenAI(api_key=api_key)

# Call the new chat completions API
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":"Say hello and your model name in one line."}],
)

print("Model reply:", resp.choices[0].message.content.strip())
