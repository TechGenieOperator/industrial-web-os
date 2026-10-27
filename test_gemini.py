import os

from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

response = llm.call("Reply with exactly: Gemini connection successful")

print(response)