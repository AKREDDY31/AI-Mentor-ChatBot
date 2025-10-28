import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- System Configuration ---

# This is the prompt that defines the bot's personality.
# We keep it on the server, not in the HTML.
SYSTEM_INSTRUCTION = """You are 'MentorBot,' a supportive and insightful AI partner. Your purpose is to act as a guide, teacher, and mentor for the user's personal and professional growth. 
Your tone is always calm, empathetic, and encouraging. 
Focus on helping the user with:
- Self-development and learning
- Self-control and discipline
- Managing mental stress and anxiety
- Finding life-work balance
- Career focus and planning

Do not be overly robotic. Be a genuine, curious partner in their wellness journey. Ask thoughtful, open-ended questions to help the user reflect, and provide actionable, practical advice. Keep your responses concise and easy to understand."""

# --- Flask App Setup ---
app = Flask(__name__)

# --- Gemini API Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=api_key)

# Initialize the Gemini Model
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-preview-09-2025',
    system_instruction=SYSTEM_INSTRUCTION
)

# --- API Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # This will send the 'ai_mentor_chatbot.html' file from the current folder.
    return send_from_directory('.', 'ai_mentor_chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat message and calls the Gemini API."""
    try:
        data = request.json
        history = data.get('history') # Full conversation history from client

        if not history:
            return jsonify({"error": "No history provided"}), 400

        # Send the entire history to Gemini
        response = model.generate_content(history)
        
        # Return only the new text reply
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"Error in /chat route: {e}")
        # Pass a useful error message back to the user
        return jsonify({"error": f"Server-side error: {str(e)}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)

