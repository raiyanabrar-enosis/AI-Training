import os
from dotenv import load_dotenv

from google import genai

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Safely retrieve the API key
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# 3. Initialize the client
client = genai.Client(api_key=api_key)

# 4. Start the chat session using your newly selected model
print("Initializing session...")
chat = client.chats.create(model="gemini-3.1-flash-lite")

print("Session started! Gemini will now remember the conversation. Type 'exit' or 'quit' to close.\n")

while True:
    # 5. Ask the user for input
    user_question = input("You: ")
    
    # Break the loop if the user wants to exit
    if user_question.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    # Skip empty inputs
    if not user_question.strip():
        continue

    # 6. Send the message to the ongoing chat session
    print("Gemini is thinking...")
    response = chat.send_message(user_question)

    # 7. Print the result
    print(f"\nGemini: {response.text}\n")