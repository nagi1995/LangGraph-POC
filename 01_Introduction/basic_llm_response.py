from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
result = llm.invoke("write a medium blog post on AI Agents with about 1000 words")

print(result)
