import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the environment variables
load_dotenv()

# Get the API key
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Print the API key
# print ("The key is: ", GROQ_API_KEY)

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1
)

# Test the LLM
print("Testing the LLM...")
print("What is your question?")
question = input("Enter your question: ")
response = llm.invoke(question)
print("The response is: ", response.content)


