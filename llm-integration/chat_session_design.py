"""

Mini Project

Build a terminal chatbot that:

Maintains conversation history
Remembers the user's name during the session
Supports an exit command
Prints both the user input and AI response
Stores history in a Python list

Once that's working, try adding:

Save the history to a JSON file when the program exits.
Load that JSON file when the program starts.
Trim the oldest messages after, say, 30 turns.

"""

import os
import json
from dotenv import load_dotenv
import datetime

from google import genai

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Safely retrieve the API key
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# 3. Initialize the client
client = genai.Client(api_key=api_key)

class ChatSession:
    def __init__(self, filename="chat_history.json"):
        self.history = []
        self.username = None
        
        # Dynamically get the directory where this script is located (project root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.filepath = os.path.join(base_dir, filename)
        
        self.load_history()

    def load_history(self):
        """Loads history from JSON if it exists, converting strings back to datetimes."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                data = json.load(f)
                for item in data:
                    if "timestamp" in item:
                        item["timestamp"] = datetime.datetime.fromisoformat(item["timestamp"])
                self.history = data
            print(f"Loaded {len(self.history)} previous messages from: {self.filepath}\n")

    def save_history(self):
        """Sorts by timestamp, keeps the last 30, and saves to JSON."""
        sorted_history = sorted(self.history, key=lambda x: x["timestamp"])
        recent_history = sorted_history[-30:]

        serializable_history = []
        for item in recent_history:
            item_copy = item.copy()
            item_copy["timestamp"] = item_copy["timestamp"].isoformat()
            serializable_history.append(item_copy)

        with open(self.filepath, "w") as f:
            json.dump(serializable_history, f, indent=4)
        print(f"History saved successfully to: {self.filepath}")

    def set_user_name(self, text):
        self.username = text
        self.add_user_message(f"The name of the user is: {text}. Please remember this.")

    def add_user_message(self, text):
        self.history.append({
            "role": "user",
            "parts": [{"text": text}],
            "timestamp": datetime.datetime.now()
        })

    def add_model_message(self, text):
        self.history.append({
            "role": "model",
            "parts": [{"text": text}],
            "timestamp": datetime.datetime.now()
        })

    def get_clean_history(self):
        wanted_keys = ['role', 'parts']
        history = [{key: historyitem[key] for key in wanted_keys if key in historyitem} for historyitem in self.history]
        return history


session = ChatSession()
print("Session started! Gemini will now remember the conversation. Type 'exit' or 'quit' to close.\n")

if not session.history:
    username = input("Please enter your name: ")
    session.set_user_name(username)

while True:
    question = input("You: ")
    
    if question.strip().lower() in ['exit', 'quit']:
        session.save_history() 
        print("Goodbye!")
        break

    if not question.strip():
        continue

    session.add_user_message(question)

    print("\nGemini: ", end="", flush=True)

    # 1. Create response stream
    response_stream = client.interactions.create(
        model="gemini-3.5-flash",
        input=question,
        stream=True
    )

    full_response = ""

    # 2. Loop through the stream chunks
    for event in response_stream:
        if hasattr(event, 'delta') and hasattr(event.delta, 'text') and event.delta.text:
            print(event.delta.text, end="", flush=True)   
            full_response += event.delta.text       

    print("\n") 

    # 3. Save the fully assembled string into your history
    session.add_model_message(full_response)